from lightning.pytorch import LightningDataModule
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple

from src.data.dataset import AortaDataset
from src.data.transforms import Compose, NormalizeHu, ConvertTypes, RandomCropPad
from src.data.constants import SPLIT_TO_NAMES, MIN_HU, MAX_HU


class AortaDataModule(LightningDataModule):
    def __init__(
        self,
        data_dirpath: Path,
        image_size: Tuple[int, int, int] = (128, 128, 32),
        num_samples: int = None,
        debug: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

    def build_trainsforms(self) -> None:
        self.train_transform = Compose(
            [
                RandomCropPad(self.hparams.image_size),
                ConvertTypes(),
                NormalizeHu(sub=MIN_HU, div=MAX_HU-MIN_HU, clip=True),
            ]
        )
        self.val_transform = self.test_transform = Compose(
            [
                ConvertTypes(),
                NormalizeHu(sub=MIN_HU, div=MAX_HU-MIN_HU, clip=True),
            ]
        )

    def setup(self, stage: str = None) -> None:
        self.build_trainsforms()
        if stage in ['fit', 'validate'] and (self.train_dataset is None or self.val_dataset is None):
            if stage == 'fit' and self.train_dataset is None:
                self.train_dataset = AortaDataset(
                    data_dirpath=self.hparams.data_dirpath,
                    names=SPLIT_TO_NAMES['train'] if not self.hparams.debug else SPLIT_TO_NAMES['train'][:5],
                    transform=self.train_transform,
                )
            if stage in ['fit', 'validate'] and self.val_dataset is None:
                self.val_dataset = AortaDataset(
                    data_dirpath=self.hparams.data_dirpath,
                    names=SPLIT_TO_NAMES['valid'],
                    transform=self.val_transform,
                    patch_size=self.hparams.image_size,
                )
        elif stage == 'test' and self.test_dataset is None:
            self.test_dataset = AortaDataset(
                data_dirpath=self.hparams.data_dirpath,
                names=SPLIT_TO_NAMES['test'],
                transform=self.test_transform,
                patch_size=self.hparams.image_size,
            )
        
    def train_dataloader(self) -> DataLoader:
        num_samples = self.hparams.num_samples
        if num_samples is None:
            num_samples = self.train_dataset.n_samples(self.hparams.image_size)
        sampler = RandomSampler(
            data_source=self.train_dataset,
            replacement=True,
            num_samples=num_samples,
        )
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=AortaDataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=None,
            shuffle=False,
            drop_last=False,
            collate_fn=AortaDataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            sampler=None,
            shuffle=False,
            drop_last=False,
            collate_fn=AortaDataset.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
