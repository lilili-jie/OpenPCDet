"""Photometric augmentation used by the full NP+ paper recipe."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    """Gaussian blur augmentation copied from the NP+ appendix recipe."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = tuple(float(value) for value in sigma)

    def __call__(self, image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarize:
    def __call__(self, image):
        return ImageOps.solarize(image)


class NPPlusPhotometricAugmentation:
    """Apply the four independent image transforms specified for NP+."""

    def __init__(self, config):
        jitter = config.get('COLOR_JITTER', [0.4, 0.4, 0.2, 0.1])
        self.color_jitter = transforms.ColorJitter(*jitter)
        self.color_jitter_probability = float(
            config.get('COLOR_JITTER_PROBABILITY', 0.8)
        )
        self.grayscale_probability = float(
            config.get('GRAYSCALE_PROBABILITY', 0.3)
        )
        self.blur_probability = float(config.get('BLUR_PROBABILITY', 0.3))
        self.solarize_probability = float(
            config.get('SOLARIZE_PROBABILITY', 0.3)
        )
        self.blur = GaussianBlur(config.get('BLUR_SIGMA', [0.1, 2.0]))
        self.solarize = Solarize()
        # The CCF-compatible loader stores OpenCV BGR arrays in PIL objects
        # and converts BGR to RGB only during normalization.  Photometric
        # transforms must see real RGB, then restore BGR for that later step.
        self.ccf_compatible_bgr = bool(
            config.get('CCF_COMPATIBLE_BGR', False)
        )

    def _to_rgb(self, image):
        if not self.ccf_compatible_bgr:
            return image.convert('RGB')
        bgr = np.asarray(image)
        return Image.fromarray(bgr[..., ::-1].copy(), mode='RGB')

    def _restore_input_order(self, image):
        if not self.ccf_compatible_bgr:
            return image
        rgb = np.asarray(image)
        return Image.fromarray(rgb[..., ::-1].copy(), mode='RGB')

    def __call__(self, image):
        image = self._to_rgb(image)
        if random.random() < self.color_jitter_probability:
            image = self.color_jitter(image)
        if random.random() < self.grayscale_probability:
            image = transforms.functional.rgb_to_grayscale(
                image, num_output_channels=3
            )
        if random.random() < self.blur_probability:
            image = self.blur(image)
        if random.random() < self.solarize_probability:
            image = self.solarize(image)
        return self._restore_input_order(image)
