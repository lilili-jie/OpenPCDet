import pickle

import os
import copy
from pathlib import Path
import numpy as np
from skimage import io
import torch
import SharedArray
import torch.distributed as dist
import cv2
from PIL import Image

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)
        self.source_multimodal = sampler_cfg.get('SOURCE_MULTIMODAL', False)
        self.multimodal_mixup = float(sampler_cfg.get('MIXUP', 0.7))
        self.source_check_2d_collision = sampler_cfg.get(
            'CHECK_2D_COLLISION', False
        )
        self.source_collision_thresholds = sampler_cfg.get(
            'COLLISION_THR', [0.0]
        )
        self.source_collision_in_classes = bool(
            sampler_cfg.get('COLLISION_IN_CLASSES', False)
        )

        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            if not db_info_path.exists():
                assert len(sampler_cfg.DB_INFO_PATH) == 1
                sampler_cfg.DB_INFO_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_INFO_PATH']
                sampler_cfg.DB_DATA_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_DATA_PATH']
                db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH[0]
                sampler_cfg.NUM_POINT_FEATURES = sampler_cfg.BACKUP_DB_INFO['NUM_POINT_FEATURES']

            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def copy_paste_to_image_kitti(self, data_dict, crop_feat, gt_number, point_idxes=None):
        kitti_img_aug_type = 'by_depth'
        kitti_img_aug_use_type = 'annotation'

        image = data_dict['images']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        corners_lidar = box_utils.boxes_to_corners_3d(boxes3d)
        if 'depth' in kitti_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int)

        if 'reverse' in kitti_img_aug_type:
            paste_order = paste_order[::-1]

        paste_mask = -255 * np.ones(image.shape[:2], dtype=np.int)
        fg_mask = np.zeros(image.shape[:2], dtype=np.int)
        overlap_mask = np.zeros(image.shape[:2], dtype=np.int)
        depth_mask = np.zeros((*image.shape[:2], 2), dtype=np.float)
        points_2d, depth_2d = data_dict['calib'].lidar_to_img(data_dict['points'][:,:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=image.shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=image.shape[0]-1)
        points_2d = points_2d.astype(np.int)
        for _order in paste_order:
            _box2d = boxes2d[_order]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = crop_feat[_order]
            overlap_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] += \
                (paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] > 0).astype(np.int)
            paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = _order

            if 'cover' in kitti_img_aug_use_type:
                # HxWx2 for min and max depth of each box region
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],0] = corners_lidar[_order,:,0].min()
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],1] = corners_lidar[_order,:,0].max()

            # foreground area of original point cloud in image plane
            if _order < gt_number:
                fg_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = 1

        data_dict['images'] = image

        # if not self.joint_sample:
        #     return data_dict

        new_mask = paste_mask[points_2d[:,1], points_2d[:,0]]==(point_idxes+gt_number)
        if False:  # self.keep_raw:
            raw_mask = (point_idxes == -1)
        else:
            raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < gt_number)
            raw_bg = (fg_mask == 0) & (paste_mask < 0)
            raw_mask = raw_fg[points_2d[:,1], points_2d[:,0]] | raw_bg[points_2d[:,1], points_2d[:,0]]
        keep_mask = new_mask | raw_mask
        data_dict['points_2d'] = points_2d

        if 'annotation' in kitti_img_aug_use_type:
            data_dict['points'] = data_dict['points'][keep_mask]
            data_dict['points_2d'] = data_dict['points_2d'][keep_mask]
        elif 'projection' in kitti_img_aug_use_type:
            overlap_mask[overlap_mask>=1] = 1
            data_dict['overlap_mask'] = overlap_mask
            if 'cover' in kitti_img_aug_use_type:
                data_dict['depth_mask'] = depth_mask

        return data_dict

    def collect_image_crops_kitti(self, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        if True:  # self.point_refine:
            # align calibration metrics for points
            points_ract = data_dict['calib'].img_to_rect(points_2d[:,0], points_2d[:,1], depth_2d)
            points_lidar = data_dict['calib'].rect_to_lidar(points_ract)
            obj_points[:, :3] = points_lidar
            # align calibration metrics for boxes
            box3d_raw = sampled_gt_boxes[idx].reshape(1,-1)
            box3d_coords = box_utils.boxes_to_corners_3d(box3d_raw)[0]
            box3d_box, box3d_depth = sampled_calib.lidar_to_img(box3d_coords)
            box3d_coord_rect = data_dict['calib'].img_to_rect(box3d_box[:,0], box3d_box[:,1], box3d_depth)
            box3d_rect = box_utils.corners_rect_to_camera(box3d_coord_rect).reshape(1,-1)
            box3d_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box3d_rect, data_dict['calib'])
            box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_rect, data_dict['calib'],
                                                                    data_dict['images'].shape[:2])
            sampled_gt_boxes[idx] = box3d_lidar[0]
            sampled_gt_boxes2d[idx] = box2d[0]

        obj_idx = idx * np.ones(len(obj_points), dtype=np.int)

        # copy crops from images
        img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        raw_image = io.imread(img_path)
        raw_image = raw_image.astype(np.float32)
        raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_boxes2d[idx].astype(np.int)
        new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int)
        raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
            new_center = new_box.reshape(2,2).mean(0)
            new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
            new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int)

        img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255

        return new_box, img_crop2d, obj_points, obj_idx

    def sample_gt_boxes_2d_kitti(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None
        # filter out box2d iou > thres
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )

        # sampled_boxes2d = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)
        boxes3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(sampled_boxes, data_dict['calib'])
        sampled_boxes2d = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, data_dict['calib'],
                                                                        data_dict['images'].shape[:2])
        sampled_boxes2d = torch.Tensor(sampled_boxes2d)
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        iou2d1 = box_utils.pairwise_iou(sampled_boxes2d, existed_boxes2d).cpu().numpy()
        iou2d2 = box_utils.pairwise_iou(sampled_boxes2d, sampled_boxes2d).cpu().numpy()
        iou2d2[range(sampled_boxes2d.shape[0]), range(sampled_boxes2d.shape[0])] = 0
        iou2d1 = iou2d1 if iou2d1.shape[1] > 0 else iou2d2

        ret_valid_mask = ((iou2d1.max(axis=1)<self.img_aug_iou_thresh) &
                         (iou2d2.max(axis=1)<self.img_aug_iou_thresh) &
                         (valid_mask))

        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        if mv_height is not None:
            mv_height = mv_height[ret_valid_mask]
        return sampled_boxes2d, mv_height, ret_valid_mask

    def sample_gt_boxes_2d(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None

        if self.img_aug_type == 'kitti':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_kitti(data_dict, sampled_boxes, valid_mask)
        else:
            raise NotImplementedError

        return sampled_boxes2d, mv_height, ret_valid_mask

    def initilize_image_aug_dict(self, data_dict, gt_boxes_mask):
        img_aug_gt_dict = None
        if self.img_aug_type is None:
            pass
        elif self.img_aug_type == 'kitti':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int)
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int)
            gt_crops2d = [data_dict['images'][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        else:
            raise NotImplementedError

        return img_aug_gt_dict

    def collect_image_crops(self, img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        if self.img_aug_type == 'kitti':
            new_box, img_crop2d, obj_points, obj_idx = self.collect_image_crops_kitti(info, data_dict,
                                                    obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx)
            img_aug_gt_dict['crop_boxes2d'].append(new_box)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
            img_aug_gt_dict['obj_index_list'].append(obj_idx)
        else:
            raise NotImplementedError

        return img_aug_gt_dict, obj_points

    def copy_paste_to_image(self, img_aug_gt_dict, data_dict, points):
        if self.img_aug_type == 'kitti':
            obj_points_idx = np.concatenate(img_aug_gt_dict['obj_index_list'], axis=0)
            point_idxes = -1 * np.ones(len(points), dtype=np.int)
            point_idxes[:obj_points_idx.shape[0]] = obj_points_idx

            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_kitti(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'], point_idxes)
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
        else:
            raise NotImplementedError
        return data_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        original_boxes2d = data_dict.get('_gt_boxes2d_for_sampling', None)
        if original_boxes2d is None:
            original_boxes2d = np.zeros((0, 5), dtype=np.float32)
        else:
            original_boxes2d = np.asarray(original_boxes2d)[gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        # convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)

        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)

            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx
                )

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        if self.source_multimodal:
            data_dict = self.paste_source_multimodal_patches(
                data_dict, total_valid_sampled_dict,
                original_boxes3d=gt_boxes,
                original_boxes2d=original_boxes2d,
            )

        if self.img_aug_type is not None:
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)

        return data_dict

    @staticmethod
    def _sequential_valid_indices(collision_matrix, num_existing):
        """Reproduce MMDataBaseSamplerV2's ordered collision resolution.

        A rejected candidate is removed from the collision matrix so a later
        candidate that collided only with it can still be retained.
        """
        collision_matrix = np.asarray(collision_matrix, dtype=np.bool_).copy()
        valid = []
        for index in range(num_existing, collision_matrix.shape[0]):
            if collision_matrix[index].any():
                collision_matrix[index, :] = False
                collision_matrix[:, index] = False
            else:
                valid.append(index - num_existing)
        return np.asarray(valid, dtype=np.int64)

    def _source_multimodal_valid_indices(
            self, sampled_dict, avoid_boxes3d, avoid_boxes2d):
        """Apply the reference sampler's combined 3-D/2-D collision test."""
        sampled_boxes3d = np.stack([
            record['box3d_lidar'] for record in sampled_dict
        ], axis=0).astype(np.float32)
        sampled_boxes2d = np.stack([
            record['box2d_camera'] for record in sampled_dict
        ], axis=0).astype(np.float32)

        avoid_boxes3d = np.asarray(avoid_boxes3d, dtype=np.float32)
        avoid_boxes2d = np.asarray(avoid_boxes2d, dtype=np.float32)
        total_boxes3d = np.concatenate(
            [avoid_boxes3d[:, :7], sampled_boxes3d[:, :7]], axis=0
        )
        total_boxes2d = np.concatenate(
            [avoid_boxes2d, sampled_boxes2d], axis=0
        )

        # Positive rotated BEV IoU is equivalent to polygon overlap for the
        # non-degenerate nuScenes boxes used by the reference collision test.
        collision = iou3d_nms_utils.boxes_bev_iou_cpu(
            total_boxes3d, total_boxes3d
        ) > 0.0

        if self.source_check_2d_collision:
            threshold = self.source_collision_thresholds
            if isinstance(threshold, (list, tuple)):
                threshold = float(np.random.choice(threshold))
            else:
                threshold = float(threshold)

            left = np.maximum(total_boxes2d[:, None, 0],
                              total_boxes2d[None, :, 0])
            top = np.maximum(total_boxes2d[:, None, 1],
                             total_boxes2d[None, :, 1])
            right = np.minimum(total_boxes2d[:, None, 2],
                               total_boxes2d[None, :, 2])
            bottom = np.minimum(total_boxes2d[:, None, 3],
                                total_boxes2d[None, :, 3])
            intersection = np.maximum(0.0, right - left) * np.maximum(
                0.0, bottom - top
            )
            if threshold == 0.0:
                collision_2d = intersection > 0.0
            else:
                # Reference iou_jit(..., mode='iof') divides each row by the
                # area of that row's foreground box.
                area = np.maximum(
                    1e-6,
                    (total_boxes2d[:, 2] - total_boxes2d[:, 0]) *
                    (total_boxes2d[:, 3] - total_boxes2d[:, 1]),
                )
                collision_2d = intersection / area[:, None] > threshold
            collision |= collision_2d

        np.fill_diagonal(collision, False)
        return self._sequential_valid_indices(
            collision, len(avoid_boxes3d)
        ), sampled_boxes3d

    @staticmethod
    def _transform_source_box2d(box2d, process_info, image_shape):
        resize, crop, _, _ = process_info
        x1, y1, x2, y2 = np.asarray(box2d, dtype=np.float32)[:4]
        x1 = int(np.floor(x1 * resize - crop[0]))
        y1 = int(np.floor(y1 * resize - crop[1]))
        x2 = int(np.ceil(x2 * resize - crop[0]))
        y2 = int(np.ceil(y2 * resize - crop[1]))
        height, width = image_shape[:2]
        clipped = (
            max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        )
        return (x1, y1, x2, y2), clipped

    def paste_source_multimodal_patches(
            self, data_dict, sampled_records, original_boxes3d,
            original_boxes2d):
        """Paste Source-only GT image crops into the matching camera view.

        Camera images have already undergone resize/crop in this OpenPCDet
        pipeline, so the stored original-resolution box is transformed by the
        target sample's per-view resize/crop parameters before blending.
        """
        images = data_dict['camera_imgs']
        origin_images = [np.asarray(image).copy() for image in images]
        process_infos = data_dict['img_process_infos']
        pasted = 0
        events = []
        for box3d, box2d in zip(original_boxes3d, original_boxes2d):
            events.append((float(box3d[0]), False, box2d, None))
        for record in sampled_records:
            events.append((float(record['box3d_lidar'][0]), True,
                           record['box2d_camera'], record))

        # Same far-to-near order as np.argsort(-all_bboxes_3d[:, 0]) in the
        # reference implementation.  Real objects participate so their
        # original pixels can restore foreground occlusion after GT pasting.
        for _, is_sampled, raw_box2d, record in sorted(
                events, key=lambda item: -item[0]):
            box2d = np.asarray(raw_box2d, dtype=np.float32)
            camera_index = int(box2d[4])
            if camera_index < 0 or camera_index >= len(images):
                continue
            target = np.asarray(images[camera_index]).copy()
            (x1, y1, x2, y2), clipped = self._transform_source_box2d(
                box2d, process_infos[camera_index], target.shape
            )
            clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped
            if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
                continue

            if not is_sampled:
                original = origin_images[camera_index]
                current_region = target[
                    clipped_y1:clipped_y2, clipped_x1:clipped_x2
                ].astype(np.float32)
                original_region = original[
                    clipped_y1:clipped_y2, clipped_x1:clipped_x2
                ].astype(np.float32)
                restored = self.multimodal_mixup * original_region
                restored += (1.0 - self.multimodal_mixup) * current_region
                target[clipped_y1:clipped_y2,
                       clipped_x1:clipped_x2] = np.clip(
                    restored, 0, 255
                ).astype(np.uint8)
                images[camera_index] = Image.fromarray(target)
                continue

            patch_path = Path(record['image_path'])
            if not patch_path.is_absolute():
                patch_path = self.root_path / patch_path
            patch = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
            if patch is None:
                raise FileNotFoundError(patch_path)

            full_width, full_height = max(1, x2 - x1), max(1, y2 - y1)
            patch = cv2.resize(
                patch, (full_width, full_height), interpolation=cv2.INTER_LINEAR
            )
            patch = patch[
                clipped_y1 - y1:clipped_y2 - y1,
                clipped_x1 - x1:clipped_x2 - x1,
            ]
            region = target[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
            paste_height, paste_width = region.shape[:2]
            mask = np.zeros((paste_height, paste_width), dtype=np.float32)
            margin_h, margin_w = int(0.05 * paste_height), int(0.05 * paste_width)
            mask[margin_h:paste_height - margin_h or paste_height,
                 margin_w:paste_width - margin_w or paste_width] = 1.0
            alpha = mask[..., None] * self.multimodal_mixup
            blended = region.astype(np.float32) * (1.0 - alpha)
            blended += patch.astype(np.float32) * alpha
            target[clipped_y1:clipped_y2, clipped_x1:clipped_x2] = np.clip(
                blended, 0, 255
            ).astype(np.uint8)
            images[camera_index] = Image.fromarray(target)
            pasted += 1
        data_dict['camera_imgs'] = images
        data_dict['multimodal_patches_pasted'] = np.asarray(
            pasted, dtype=np.int64
        )
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        original_boxes = gt_boxes
        original_boxes2d = data_dict.get('_gt_boxes2d_for_sampling', None)
        if original_boxes2d is None:
            original_boxes2d = np.zeros((0, 5), dtype=np.float32)
        avoid_boxes = original_boxes
        avoid_boxes2d = original_boxes2d
        total_valid_sampled_dict = []
        total_valid_sampled_boxes = []
        sampled_mv_height = []
        sampled_gt_boxes2d = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'
                if self.source_multimodal:
                    valid_indices, sampled_boxes = (
                        self._source_multimodal_valid_indices(
                            sampled_dict, avoid_boxes, avoid_boxes2d
                        )
                    )
                    valid_mask = np.zeros(len(sampled_dict), dtype=np.bool_)
                    valid_mask[valid_indices] = True
                else:
                    sampled_boxes = np.stack([
                        x['box3d_lidar'] for x in sampled_dict
                    ], axis=0).astype(np.float32)
                    iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(
                        sampled_boxes[:, 0:7], avoid_boxes[:, 0:7]
                    )
                    iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(
                        sampled_boxes[:, 0:7], sampled_boxes[:, 0:7]
                    )
                    np.fill_diagonal(iou2, 0)
                    iou1 = iou1 if iou1.shape[1] > 0 else iou2
                    valid_mask = (
                        (iou1.max(axis=1) + iou2.max(axis=1)) == 0
                    )

                if self.img_aug_type is not None:
                    sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, valid_mask)
                    sampled_gt_boxes2d.append(sampled_boxes2d)
                    if mv_height is not None:
                        sampled_mv_height.append(mv_height)

                valid_mask = valid_mask.nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                total_valid_sampled_dict.extend(valid_sampled_dict)
                total_valid_sampled_boxes.append(
                    valid_sampled_boxes[:, :original_boxes.shape[-1]]
                )
                if (self.source_multimodal and self.source_collision_in_classes
                        and valid_sampled_dict):
                    avoid_boxes = np.concatenate(
                        [avoid_boxes, valid_sampled_boxes[:, :avoid_boxes.shape[-1]]],
                        axis=0,
                    )
                    new_boxes2d = np.stack([
                        record['box2d_camera'] for record in valid_sampled_dict
                    ], axis=0)
                    avoid_boxes2d = np.concatenate(
                        [avoid_boxes2d, new_boxes2d], axis=0
                    )

        sampled_gt_boxes = (
            np.concatenate(total_valid_sampled_boxes, axis=0)
            if total_valid_sampled_boxes
            else np.zeros((0, original_boxes.shape[-1]), dtype=np.float32)
        )

        if total_valid_sampled_dict.__len__() > 0:
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )

        data_dict.pop('gt_boxes_mask')
        data_dict.pop('_gt_boxes2d_for_sampling', None)
        return data_dict
