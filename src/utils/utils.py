import gc
import torch
import numpy as np
import scipy
from medpy.io import save
from typing import Dict, Optional, Union, Any, Tuple
from lightning import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from weakref import proxy


###################################################################
########################## General Utils ##########################
###################################################################

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        """Add argument links to parser.

        Example:
            parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        """
        return

    def before_instantiate_classes(self) -> None:
        # Set LR: nested dict value setting from CLI is not supported
        # so separate arg is used
        if 'fit' in self.config and self.config['fit']['model']['init_args']['lr'] is not None:
            self.config['fit']['model']['init_args']['optimizer_init']['init_args']['lr'] = \
                self.config['fit']['model']['init_args']['lr']



class TrainerWandb(Trainer):
    """Hotfix for wandb logger saving config & artifacts to project root dir
    and not in experiment dir."""
    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if isinstance(self.loggers[0], WandbLogger) and self.loggers[0]._experiment is not None:
                dirpath = self.loggers[0]._experiment.dir
            elif not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath


class ModelCheckpointNoSave(ModelCheckpoint):
    def best_epoch(self) -> int:
        # exmple: epoch=10-step=1452.ckpt
        return int(self.best_model_path.split('=')[-2].split('-')[0])
    
    def ith_epoch_score(self, i: int) -> Optional[float]:
        # exmple: epoch=10-step=1452.ckpt
        ith_epoch_filepath_list = [
            filepath 
            for filepath in self.best_k_models.keys()
            if f'epoch={i}-' in filepath
        ]
        
        # Not found
        if not ith_epoch_filepath_list:
            return None
    
        ith_epoch_filepath = ith_epoch_filepath_list[-1]
        return self.best_k_models[ith_epoch_filepath]

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


class TempSetContextManager:
    def __init__(self, obj, attr, value):
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self):
        self.old_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.old_value)



def state_norm(module: torch.nn.Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each state dict tensor's norm and their overall norm.

    The overall norm is computed over all tensor together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the tensor norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the tensor viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"state_{norm_type}_norm{group_separator}{name}": p.data.float().norm(norm_type)
        for name, p in module.state_dict().items()
        if not 'num_batches_tracked' in name
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"state_{norm_type}_norm_total"] = total_norm
    return norms


# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/26
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes, dtype=torch.bool, device=labels.device) 
    return y[labels] 


# https://github.com/bnsreenu/python_for_microscopists/blob/master/
# 229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
def spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2*(scipy.signal.windows.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.windows.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def spline_window_3d(h, w, d, power=2):
    h_wind = spline_window(h, power)
    w_wind = spline_window(w, power)
    d_wind = spline_window(d, power)
    return h_wind[:, None, None] * w_wind[None, :, None] * d_wind[None, None, :] 


class UnpatchifyMetrics:
    """Unpatchify predictions to original full image assuming the dataloader is sequential
    and calculate the metrics for each full image, then aggregate them.
    """
    def __init__(self, n_classes, metrics, save_dirpath):
        self.n_classes = n_classes
        self.metrics = metrics
        self.save_dirpath = save_dirpath
        if self.save_dirpath is not None:
            self.save_dirpath.mkdir(parents=True, exist_ok=True)
        self.name = None
        self.preds = None
        self.weigths = None
        self.masks = None
        self.original_shape = None 
        self.weight_kernel = None

    def _reset_current(self):
        self.name = None
        self.preds = None
        self.weights = None
        self.masks = None
        self.original_shape = None 
        self.weight_kernel = None

        # Cleanup heavy tensors
        gc.collect()

    def reset(self):
        self._reset_current()
        for metric in self.metrics.values():
            metric.reset()

    def _calculate_metrics(self):
        # Remove padding
        self.preds = self.preds[
            :,
            :self.original_shape[0], 
            :self.original_shape[1], 
            :self.original_shape[2]
        ]
        self.masks = self.masks[
            :self.original_shape[0], 
            :self.original_shape[1], 
            :self.original_shape[2]
        ]

        # Weighted average
        self.weights[self.weights == 0] = 1
        self.preds /= self.weights

        # One-hot
        self.preds = torch.argmax(self.preds, dim=0)
        self.preds = one_hot_embedding(
            self.preds, 
            num_classes=self.n_classes
        ).permute(3, 0, 1, 2).unsqueeze(0)
        self.masks = one_hot_embedding(
            self.masks, 
            num_classes=self.n_classes
        ).permute(3, 0, 1, 2).unsqueeze(0)

        # Save predictions
        if self.save_dirpath is not None:
            preds_to_save = self.preds.cpu().numpy().astype(np.int16)
            save(preds_to_save, self.save_dirpath / f'{self.name}.mha', use_compression=True)

        # Calculate metrics
        for metric in self.metrics.values():
            metric(self.preds, self.masks)

    def _init(
        self, 
        name: str, 
        padded_shape: Tuple[int, int, int], 
        original_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int, int],
        device: torch.device = torch.device('cpu'),
    ):
        self.name = name
        self.preds = torch.zeros((self.n_classes, *padded_shape), dtype=torch.float32, device=device)
        self.weights = torch.zeros((self.n_classes, *padded_shape), dtype=torch.float32, device=device)
        self.masks = torch.zeros(padded_shape, dtype=torch.int32, device=device)
        self.original_shape = original_shape
        self.weight_kernel = torch.from_numpy(
            spline_window_3d(*patch_shape, power=2)
        )

    def update(self, batch: Dict[str, Any]):
        """Update the metrics with the predictions and masks of the batch.
        batch:
            - 'name': List[str]
            - 'padded_shape': List[Tuple[int, int, int]]
            - 'original_shape': List[Tuple[int, int, int]]
            - 'indices': List[Tensor[B, 3, 2]]
            - 'pred': FloatTensor[B, C, H, W, D], probabilities
            - 'mask': LongTensor[B, H, W, D], ground truth
        """
        batch['mask'] = batch['mask'].to(torch.int32)

        B = batch['mask'].shape[0]
        for i in range(B):
            name: str = batch['name'][i]
            if self.name is None or name != self.name:
                if self.name is not None:
                    self._calculate_metrics()
                    self._reset_current()
                patch_shape = (
                    batch['indices'][i][0][1] - batch['indices'][i][0][0],
                    batch['indices'][i][1][1] - batch['indices'][i][1][0],
                    batch['indices'][i][2][1] - batch['indices'][i][2][0],
                )
                self._init(
                    name, 
                    batch['padded_shape'][i], 
                    batch['original_shape'][i],
                    patch_shape,
                    device=batch['mask'].device,
                )
            self.preds[
                :,
                batch['indices'][i][0][0]:batch['indices'][i][0][1],
                batch['indices'][i][1][0]:batch['indices'][i][1][1],
                batch['indices'][i][2][0]:batch['indices'][i][2][1],
            ] += (batch['pred'][i] * self.weight_kernel[None, ...])
            self.weights[
                :,
                batch['indices'][i][0][0]:batch['indices'][i][0][1],
                batch['indices'][i][1][0]:batch['indices'][i][1][1],
                batch['indices'][i][2][0]:batch['indices'][i][2][1],
            ] += self.weight_kernel[None, ...]
            self.masks[
                batch['indices'][i][0][0]:batch['indices'][i][0][1],
                batch['indices'][i][1][0]:batch['indices'][i][1][1],
                batch['indices'][i][2][0]:batch['indices'][i][2][1],
            ] = batch['mask'][i]

    def compute(self):
        return {
            name: metric.aggregate().item()
            for name, metric in self.metrics.items()
        }
