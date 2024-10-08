from lightning.pytorch import LightningDataModule
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple
from volumentations import RotatePseudo2D, GridDistortion

from src.data.dataset import AortaDataset
from src.data.transforms import Compose, NormalizeHu, ConvertTypes, RandomCropPad, CenteredGaussianNoise
from src.data.constants import SPLIT_TO_NAMES, MIN_HU, MAX_HU


class AortaDataModule(LightningDataModule):
    def __init__(
        self,
        data_dirpath: Path,
        image_size: Tuple[int, int, int] = (128, 128, 32),
        num_samples: int = None,
        val_test_crop_inv_share: int = 10,
        only_train: bool = False,
        fold_index: int = 0,
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

        all_names = SPLIT_TO_NAMES['train'] + SPLIT_TO_NAMES['valid'] + SPLIT_TO_NAMES['test']
        if only_train:
            self.split_to_names = {
                'train': all_names,
                'valid': [],
                'test': [],
            }
        else:
            assert fold_index in range(5)
            kfold = KFold(n_splits=5, shuffle=True, random_state=28490467)
            indices = list(kfold.split(all_names))
            self.split_to_names = {
                'train': [all_names[i] for i in indices[fold_index][0]],
                'valid': [all_names[i] for i in indices[fold_index][1]],
                'test': [],
            }

    def build_trainsforms(self) -> None:
        self.train_transform = Compose(
            [
                RandomCropPad(self.hparams.image_size),
                ConvertTypes(),
                CenteredGaussianNoise(p=0.5), 
                GridDistortion(p=0.5), 
                RotatePseudo2D(p=0.5), 
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
                    names=self.split_to_names['train'] if not self.hparams.debug else self.split_to_names['train'][:5],
                    transform=self.train_transform,
                    pad_size=self.hparams.image_size,
                    # Omit background by mask
                    cbp_margin=10,
                    crop_inv_share=None,
                )
            if stage in ['fit', 'validate'] and self.val_dataset is None:
                self.val_dataset = AortaDataset(
                    data_dirpath=self.hparams.data_dirpath,
                    names=self.split_to_names['valid'],
                    transform=self.val_transform,
                    patch_size=self.hparams.image_size,
                    pad_size=self.hparams.image_size,
                    # Omit fixed background by share of size
                    cbp_margin=None,
                    crop_inv_share=self.hparams.val_test_crop_inv_share,
                )
        elif stage == 'test' and self.test_dataset is None:
            # TODO: undo cropping and padding for test dataset
            self.test_dataset = AortaDataset(
                data_dirpath=self.hparams.data_dirpath,
                names=self.split_to_names['test'],
                transform=self.test_transform,
                patch_size=self.hparams.image_size,
                pad_size=self.hparams.image_size,
                # Omit fixed background by share of size
                cbp_margin=None,
                crop_inv_share=self.hparams.val_test_crop_inv_share,
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
