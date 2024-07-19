import numpy as np
from patchify import patchify
from medpy import io
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from torch.utils.data import default_collate


def generate_patches_3d(
    *arrays, 
    patch_size: Tuple[int, int, int] | None = None, 
    step_size: int = None,
):
    assert all(arr.ndim == 3 for arr in arrays)
    assert all(arr.shape == arrays[0].shape for arr in arrays)
    shape_before_padding = arrays[0].shape

    if patch_size is None:
        yield arrays, (0, 0, 0), shape_before_padding, shape_before_padding, (1, 1, 1)
    else:
        # Pad the arrays to make them divisible by the patch size
        arrays = [
            np.pad(
                arr, 
                [
                    (0, patch_size[i] - arr.shape[i] % patch_size[i]) 
                    for i in range(3)
                ]
            ) 
            for arr in arrays
        ]
        shape_after_padding = arrays[0].shape

        # Patchify the arrays
        arrays = [
            patchify(arr, patch_size, step_size)
            for arr in arrays
        ]
        shape_patches = arrays[0].shape[:3]

        # Yield the patches
        for i in range(shape_patches[0]):
            for j in range(shape_patches[1]):
                for k in range(shape_patches[2]):
                    yield tuple(
                        arr[i, j, k, ...] 
                        for arr in arrays
                    ), (i, j, k), shape_before_padding, shape_after_padding, shape_patches


class AortaDataset:
    def __init__(
        self, 
        data_dirpath: Path, 
        names: List[str],
        transform: Optional[Callable] = None,
        patch_size: Tuple[int, int, int] | None = None, 
    ):
        assert patch_size is None or all(p == patch_size[0] for p in patch_size)
        step_size = None if patch_size is None else patch_size[0]

        self.data = []
        for name in names:
            image, _ = io.load(data_dirpath / 'images' / f'subject{name:03}_CTA.mha')
            mask, _ = io.load(data_dirpath / 'masks' / f'subject{name:03}_label.mha')
            for (
                (image_patch, mask_patch), 
                patch_indices, 
                shape_before_padding, 
                shape_after_padding,
                shape_patches
            ) in generate_patches_3d(
                image, mask, patch_size=patch_size, step_size=step_size,
            ):
                self.data.append(
                    {
                        'image': image_patch,
                        'mask': mask_patch,
                        'name': name,
                        'patch_indices': patch_indices,
                        'shape_before_padding': shape_before_padding,
                        'shape_after_padding': shape_after_padding,
                        'shape_patches': shape_patches,
                    }
                )
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
            if key in ['name', 'patch_indices', 'shape_before_padding', 'shape_after_padding', 'shape_patches']:
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