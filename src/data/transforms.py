import numpy as np
import random

from src.data.constants import TOTAL_POSITIVE_COUNTERS, INDIVIDUAL_POSITIVE_COUNTERS, CLASSES


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data):
        # Hack with force_apply and targets to work with volumentations
        for t in self.transforms:
            data = t(force_apply=False, targets=['image', 'mask'], **data)
            data.pop('force_apply', None)
            data.pop('targets', None)
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
    def __init__(self, shape, weighted=False):
        self.shape = shape
        self.weighted = weighted

    def __call__(self, **data):
        if not self.weighted:
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
        else:
            # Get weights: only consider classes that are present in the current mask
            present_classes_mask = INDIVIDUAL_POSITIVE_COUNTERS[data['name']] > 0
            classes = CLASSES[present_classes_mask]
            weights = TOTAL_POSITIVE_COUNTERS[present_classes_mask]
            weights = 1 / weights
            weights /= weights.sum()

            # Get random class
            random_class = random.choices(classes, weights)[0]

            # Get center as random indices inside the mask of selected class
            mask_to_sample_from = data['mask'] == random_class
            nonzero = np.nonzero(mask_to_sample_from)
            n_nonzero = len(nonzero[0])
            random_nonzero_index = random.randint(0, n_nonzero - 1)
            h_center, w_center, d_center = [nonzero[i][random_nonzero_index] for i in range(3)]

            # Get crop indices
            h_start = max(0, h_center - self.shape[0] // 2)
            w_start = max(0, w_center - self.shape[1] // 2)
            d_start = max(0, d_center - self.shape[2] // 2)
            h_stop = min(h_start + self.shape[0], data['image'].shape[0])
            w_stop = min(w_start + self.shape[1], data['image'].shape[1])
            d_stop = min(d_start + self.shape[2], data['image'].shape[2])

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


class RandomFillPlane:
    def __init__(self, axis, p):
        self.axis = axis
        self.p = p

    def __call__(self, **data):
        drop_mask = np.random.rand(data['image'].shape[self.axis]) < self.p
        if self.axis == 0:
            data['image'][drop_mask, :, :] = 0
            data['mask'][drop_mask, :, :] = 0
        elif self.axis == 1:
            data['image'][:, drop_mask, :] = 0
            data['mask'][:, drop_mask, :] = 0
        elif self.axis == 2:
            data['image'][:, :, drop_mask] = 0
            data['mask'][:, :, drop_mask] = 0
        return data


class CenteredGaussianNoise:
    def __init__(self, std_lim=(0, 100), p=0.5):
        self.std_lim = std_lim
        self.p = p

    def __call__(self, **data):
        if random.random() < self.p:
            std = random.uniform(*self.std_lim)
            noise = np.random.normal(0, std, data['image'].shape)
            data['image'] += noise
        return data
