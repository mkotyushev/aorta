import numpy as np
from patchify import patchify
from medpy import io
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from torch.utils.data import default_collate
from scipy.ndimage import distance_transform_edt

from src.data.constants import N_CLASSES


def crop_possiby_padded(arr, h_start, h_stop, w_start, w_stop, d_start, d_stop, patch_size):
    # Crop the array, but make sure that the indices are within the array
    arr = arr[
        ...,
        h_start:min(h_stop, arr.shape[-3]),
        w_start:min(w_stop, arr.shape[-2]),
        d_start:min(d_stop, arr.shape[-1]),
    ]

    # Pad if necessary
    if (
        arr.shape[-3] < patch_size[0] or 
        arr.shape[-2] < patch_size[1] or 
        arr.shape[-1] < patch_size[2]
    ):
        padding = [(0, 0)] * (arr.ndim - 3) + [
            (0, patch_size[0] - arr.shape[0]),
            (0, patch_size[1] - arr.shape[1]),
            (0, patch_size[2] - arr.shape[2]),
        ]
        arr = np.pad(
            arr,
            padding,
            mode='constant',
            constant_values=0,
        )

    return arr 


def generate_patches_3d(
    *arrays, 
    patch_size: Tuple[int, int, int] | None = None, 
    step_size: int = None,
):
    assert all(arr.shape[-3:] == arrays[0].shape[-3:] for arr in arrays)
    original_shape = arrays[0].shape[-3:]

    if patch_size is None:
        yield arrays, (0, 0, 0), original_shape, original_shape
    else:
        if step_size is None:
            step_size = patch_size
        
        padded_shape = []
        for i in range(3):
            if (original_shape[i] - patch_size[i]) % step_size[i] == 0:
                padded_shape.append(original_shape[i])
            else:
                padded_shape.append(
                    (original_shape[i] - patch_size[i]) // step_size[i] * step_size[i] + patch_size[i]
                )
        padded_shape = tuple(padded_shape)

        for h_start in range(0, padded_shape[0] - patch_size[0] + 1, step_size[0]):
            h_stop = h_start + patch_size[0]
            for w_start in range(0, padded_shape[1] - patch_size[1] + 1, step_size[1]):
                w_stop = w_start + patch_size[1]
                for d_start in range(0, padded_shape[2] - patch_size[2] + 1, step_size[2]):
                    d_stop = d_start + patch_size[2]

                    indices = (
                        (h_start, h_stop),
                        (w_start, w_stop),
                        (d_start, d_stop),
                    )
                    
                    image_patches = [
                        crop_possiby_padded(
                            arr, 
                            h_start, h_stop, 
                            w_start, w_stop, 
                            d_start, d_stop, 
                            patch_size
                        )
                        for arr in arrays
                    ]
                    
                    yield image_patches, indices, original_shape, padded_shape


def crop_by_positive(mask, image=None, dtm=None, margin=0, pad_size=None):
    assert mask.ndim == 3
    shape = mask.shape
    if image is not None:
        assert image.ndim == 3
        assert image.shape == shape
    if dtm is not None:
        assert dtm.ndim == 4
        assert dtm.shape[-3:] == shape 

    is_bg = (mask == 0)

    is_bg_h = is_bg.all((1, 2))
    is_bg_w = is_bg.all((0, 2))
    is_bg_d = is_bg.all((0, 1))

    h_start, h_stop = is_bg_h.argmin(), len(is_bg_h) - is_bg_h[::-1].argmin()
    w_start, w_stop = is_bg_w.argmin(), len(is_bg_w) - is_bg_w[::-1].argmin()
    d_start, d_stop = is_bg_d.argmin(), len(is_bg_d) - is_bg_d[::-1].argmin()

    h_start = max(0, h_start - margin)
    h_stop = min(shape[0], h_stop + margin)
    w_start = max(0, w_start - margin)
    w_stop = min(shape[1], w_stop + margin)
    d_start = max(0, d_start - margin)
    d_stop = min(shape[2], d_stop + margin)

    if image is not None:
        image = image[h_start:h_stop, w_start:w_stop, d_start:d_stop]
    mask = mask[h_start:h_stop, w_start:w_stop, d_start:d_stop]
    if dtm is not None:
        dtm = dtm[:, h_start:h_stop, w_start:w_stop, d_start:d_stop]

    if (
        pad_size is not None and
        (
            shape[0] < pad_size[0] or 
            shape[1] < pad_size[1] or 
            shape[2] < pad_size[2])
    ):
        h_pad = max(0, pad_size[0] - shape[0])
        w_pad = max(0, pad_size[1] - shape[1])
        d_pad = max(0, pad_size[2] - shape[2])

        h_pad_before = h_pad // 2
        h_pad_after = h_pad - h_pad_before
        w_pad_before = w_pad // 2
        w_pad_after = w_pad - w_pad_before
        d_pad_before = d_pad // 2
        d_pad_after = d_pad - d_pad_before

        pad = ((h_pad_before, h_pad_after), (w_pad_before, w_pad_after), (d_pad_before, d_pad_after))

        if image is not None:
            image = np.pad(image, pad, mode='constant', constant_values=0)
        mask = np.pad(mask, pad, mode='constant', constant_values=0)
        if dtm is not None:
            dtm = np.pad(dtm, pad, mode='constant', constant_values=0)

    return mask, image, dtm


class AortaDataset:
    def __init__(
        self, 
        data_dirpath: Path, 
        names: List[str],
        transform: Optional[Callable] = None,
        patch_size: Tuple[int, int, int] | None = None, 
        pad_size: Tuple[int, int, int] | None = None,
        load_dtm: bool = False,
    ):
        step_size = None if patch_size is None else patch_size

        self.data = []
        for name in names:
            image, _ = io.load(data_dirpath / 'images' / f'subject{name:03}_CTA.mha')
            mask, _ = io.load(data_dirpath / 'masks' / f'subject{name:03}_label.mha')
            dtm = None
            if load_dtm:
                dtm = np.load(data_dirpath / 'dtms' / f'subject{name:03}_label.npy')

            # DTM already cropped by mask when saved
            mask, image, _ = crop_by_positive(mask, image, dtm=None, margin=10, pad_size=pad_size)
            assert mask.shape == image.shape 
            if load_dtm:
                assert dtm.shape[-3:] == image.shape
            print(f'Loaded {name}, image shape: {image.shape}, mask shape: {mask.shape}, dtm shape: {None if dtm is None else dtm.shape}')

            if dtm is None:
                arrays = image, mask
            else:
                arrays = image, mask, dtm

            for (
                arrays_patch, 
                indices, 
                original_shape, 
                padded_shape
            ) in generate_patches_3d(
                *arrays, patch_size=patch_size, step_size=step_size,
            ):
                if dtm is None:
                    image_patch, mask_patch = arrays_patch
                else:
                    image_patch, mask_patch, dtm_patch = arrays_patch

                item = {
                    'image': image_patch,
                    'mask': mask_patch,
                    'name': name,
                    'indices': indices,
                    'original_shape': original_shape,
                    'padded_shape': padded_shape,
                }
                if dtm is not None:
                    item['dtm'] = dtm_patch
                self.data.append(item)

        self.patch_size = patch_size
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(**item)
        return item

    @staticmethod
    def collate_fn(batch):
        output = dict()
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if key in ['name', 'indices', 'original_shape', 'padded_shape']:
                output[key] = values
            else:
                output[key] = default_collate(values)
                if key == 'image':
                    output[key] = output[key][:, None, ...]
        return output
    
    def n_samples(self, image_size: Tuple[int, int, int]):
        if self.patch_size is not None:
            # Patchified dataset, n_samples = len(self.data)
            return self.__len__()
        # Not patchified, calculate how many patches ~ 
        # can be extracted from all the images
        n_samples = 0
        for item in self.data:
            shape = item['image'].shape
            n_samples += int(
                np.prod(
                    [
                        np.ceil(shape[i] / image_size[i])
                        for i in range(3)
                    ]
                )
            )
        return n_samples