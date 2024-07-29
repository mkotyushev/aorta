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

    
class RandomCrop:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, **data):
        h_start = random.randint(0, data['image'].shape[0] - self.shape[0])
        w_start = random.randint(0, data['image'].shape[1] - self.shape[1])
        d_start = random.randint(0, data['image'].shape[2] - self.shape[2])

        data['image'] = data['image'][h_start:h_start+self.shape[0], w_start:w_start+self.shape[1], d_start:d_start+self.shape[2]]
        data['mask'] = data['mask'][h_start:h_start+self.shape[0], w_start:w_start+self.shape[1], d_start:d_start+self.shape[2]]

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
