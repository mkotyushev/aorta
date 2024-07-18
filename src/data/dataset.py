import numpy as np
from medpy import io
from pathlib import Path
from typing import List, Optional, Callable
from torch.utils.data import default_collate


class AortaDataset:
    def __init__(
        self, 
        data_dirpath: Path, 
        names: List[str],
        transform: Optional[Callable] = None,
    ):
        self.data = []
        for name in names:
            image, _ = io.load(data_dirpath / 'images' / f'subject{name:03}_CTA.mha')
            mask, _ = io.load(data_dirpath / 'masks' / f'subject{name:03}_label.mha')
            self.data.append(
                {
                    'image': image,
                    'mask': mask,
                    'name': name,
                }
            )
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
        return {
            'image': default_collate([sample['image'] for sample in batch])[:, None, ...],
            'mask': default_collate([sample['mask'] for sample in batch]),
            'name': [sample['name'] for sample in batch],
        }