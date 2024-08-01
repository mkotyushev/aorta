import numpy as np
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data):
        for t in self.transforms:
            data = t(**data)
        return data


class ConvertTypes:
    def __call__(self, **data):
        data['image'] = data['image'].astype(np.float32)
        data['mask'] = data['mask'].astype(np.int32)
        return data


class NormalizeHu:
    def __init__(self, sub, div, clip=True):
        self.sub = sub
        self.div = div
        self.clip = clip
    
    def __call__(self, **data):
        data['image'] = (data['image'] - self.sub) / self.div
        if self.clip:
            data['image'] = np.clip(data['image'], 0.0, 1.0)
        return data


class RandomCropPad:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, **data):
        h_start, w_start, d_start = 0, 0, 0
        if data['image'].shape[0] >= self.shape[0]:
            h_start = random.randint(0, data['image'].shape[0] - self.shape[0])
        if data['image'].shape[1] >= self.shape[1]:
            w_start = random.randint(0, data['image'].shape[1] - self.shape[1])
        if data['image'].shape[2] >= self.shape[2]:
            d_start = random.randint(0, data['image'].shape[2] - self.shape[2])

        h_stop = min(h_start+self.shape[0], data['image'].shape[0])
        w_stop = min(w_start+self.shape[1], data['image'].shape[1])
        d_stop = min(d_start+self.shape[2], data['image'].shape[2])

        data['image'] = data['image'][h_start:h_stop, w_start:w_stop, d_start:d_stop]
        data['mask'] = data['mask'][h_start:h_stop, w_start:w_stop, d_start:d_stop]

        if (
            data['image'].shape[0] < self.shape[0] or
            data['image'].shape[1] < self.shape[1] or
            data['image'].shape[2] < self.shape[2]
        ):
            h_pad = max(0, self.shape[0] - data['image'].shape[0])
            w_pad = max(0, self.shape[1] - data['image'].shape[1])
            d_pad = max(0, self.shape[2] - data['image'].shape[2])

            h_pad_before = h_pad // 2
            h_pad_after = h_pad - h_pad_before
            w_pad_before = w_pad // 2
            w_pad_after = w_pad - w_pad_before
            d_pad_before = d_pad // 2
            d_pad_after = d_pad - d_pad_before

            pad = ((h_pad_before, h_pad_after), (w_pad_before, w_pad_after), (d_pad_before, d_pad_after))
            data['image'] = np.pad(data['image'], pad, mode='constant', constant_values=0)
            data['mask'] = np.pad(data['mask'], pad, mode='constant', constant_values=0)

        return data


class RandomFlip:
    def __init__(self, axis, p):
        self.axis = axis
        self.p = p

    def __call__(self, **data):
        if random.random() < self.p:
            data['image'] = np.flip(data['image'], axis=self.axis)
            data['mask'] = np.flip(data['mask'], axis=self.axis)
        return data
