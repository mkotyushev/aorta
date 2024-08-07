import cv2
import numpy as np
import random
from volumentations import Transform, to_tuple
from volumentations.augmentations import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data):
        # Hack with force_apply and targets to work with volumentations
        for t in self.transforms:
            data = t(force_apply=False, targets=['image', 'mask', 'dtm'], **data)
            data.pop('force_apply', None)
            data.pop('targets', None)
        return data


class ConvertTypes:
    def __call__(self, **data):
        data['image'] = data['image'].astype(np.float32)
        data['mask'] = data['mask'].astype(np.int32)
        if 'dtm' in data:
            # data['dtm'].shape == (classes, H, W, D)
            # data['dtm_max'].shape == (classes,)
            data['dtm'] = ((data['dtm'].astype(np.float32) / 255.0) * 2 - 1) * data['dtm_max'][:, None, None, None]
        
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
        if 'dtm' in data:
            data['dtm'] = data['dtm'][:, h_start:h_stop, w_start:w_stop, d_start:d_stop]

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
            if 'dtm' in data:
                data['dtm'] = np.pad(data['dtm'], ((0, 0), *pad), mode='constant', constant_values=0)

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


class TrialTransform(Transform):
    def __call__(self, force_apply, targets, **data):
        if force_apply or self.always_apply or random.random() < self.p:
            params = self.get_params(**data)

            for k, v in data.items():
                if k in targets[0]:
                    data[k] = self.apply(v, **params)
                elif k in targets[1]:
                    data[k] = self.apply_to_mask(v, **params)
                elif k in targets[2]:
                    data[k] = self.apply_to_dtm(v, **params)
                else:
                    data[k] = v

        return data

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)
    
    def apply_to_dtm(self, mask, **params):
        return self.apply(mask, **params)


class GridDistortion(TrialTransform):
    """
    Args:
        num_steps (int): count of grid cells on each side.
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
    Targets:
        image, mask, dtm
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        num_steps=5,
        distort_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        img_transformed = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[2]):
            img_transformed[:,:,slice] = F.grid_distortion(img[:,:,slice],
                                                           self.num_steps,
                                                           stepsx,
                                                           stepsy,
                                                           interpolation,
                                                           self.border_mode,
                                                           self.value)
        return img_transformed

    def apply_to_dtm(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        img_transformed = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[0]):
            img_transformed[slice, ...] = self.apply(img[slice, ...],
                                                           stepsx,
                                                           stepsy,
                                                           interpolation)
        return img_transformed
    
    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        img_transformed = np.zeros(img.shape, dtype=img.dtype)
        for slice in range(img.shape[2]):
            img_transformed[:,:,slice] = F.grid_distortion(img[:,:,slice],
                                                           self.num_steps,
                                                           stepsx,
                                                           stepsy,
                                                           cv2.INTER_NEAREST,
                                                           self.border_mode,
                                                           self.mask_value)
        return img_transformed

    def get_params(self, **data):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self):
        return (
            "num_steps",
            "distort_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )


class RotatePseudo2D(TrialTransform):
    def __init__(self, axes=(0,1), limit=(-90, 90), interpolation=1, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle):
        return F.rotate2d(img, angle, axes=self.axes, reshape=False, interpolation=self.interpolation, border_mode=self.border_mode, value=self.value)

    def apply_to_dtm(self, img, angle):
        return F.rotate2d(img, angle, axes=(a + 1 for a in self.axes), reshape=False, interpolation=self.interpolation, border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, mask, angle):
        return F.rotate2d(mask, angle, axes=self.axes, reshape=False, interpolation=0, border_mode=self.border_mode, value=self.mask_value)

    def get_params(self, **data):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}
