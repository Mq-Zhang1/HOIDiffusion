# -*- coding: utf-8 -*-

import cv2
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from transformers import CLIPProcessor
import torch
from basicsr.utils import img2tensor


class PILtoTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['jpg'] = to_tensor(sample['jpg'])
        if 'openpose' in sample:
            sample['openpose'] = to_tensor(sample['openpose'])
        return sample


class AddCannyFreezeThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        canny = cv2.Canny(img, self.low_threshold, self.high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        return sample


class AddCannyRandomThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200, shift_range=50):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.threshold_prng = np.random.RandomState()
        self.shift_range = shift_range

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        low_threshold = self.low_threshold + self.threshold_prng.randint(-self.shift_range, self.shift_range)
        high_threshold = self.high_threshold + self.threshold_prng.randint(-self.shift_range, self.shift_range)
        canny = cv2.Canny(img, low_threshold, high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        return sample


class AddStyle(object):

    def __init__(self, version):
        self.processor = CLIPProcessor.from_pretrained(version)
        self.pil_to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        style = self.processor(images=x, return_tensors="pt")['pixel_values'][0]
        sample['style'] = style
        return sample


class AddSpatialPalette(object):

    def __init__(self, downscale_factor=64):
        self.downscale_factor = downscale_factor

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        color = cv2.resize(img, (w // self.downscale_factor, h // self.downscale_factor), interpolation=cv2.INTER_CUBIC)
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['color'] = img2tensor(color, bgr2rgb=True, float32=True) / 255.
        return sample

from ldm.modules.midas.api import load_midas_transform


class AddMiDaS(object):
    def __init__(self, model_type):
        super().__init__()
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample['jpg'])
        x = self.transform({"image": x})["image"]
        sample['midas_in'] = x
        return sample