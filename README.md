# Introduction

This repository contains the training code for @mkotyushev's part of the LOWER MATH team solution for [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/) (team's final test phase score is [TBA], securing [TBA] place). The @rostepifanov's part of the solution could be found in [TBA], and the common inference code could be found in [this](https://github.com/mkotyushev/aorta_submission/) repository.

# Solution overview

The solution is based on the U-Net++ architecture with the following modifications:
- Model is converted to 3D by replacing 2D convolutions with 3D ones and modifying the architecture accordingly
- Additional heads are added along with default U-Net++ head, accumulating activations from lower levels of the network

See more details on training process, used augmentations & other hyperparameters in the [TBA] report paper.

@mkotyushev's models training is done using 5-fold cross-validation, the best model by epoch is selected for each fold and their probabilities are averaged. Due to inference time restriction, only two models are used in the final submission from @mkotyushev in contrast to 4 models from @rostepifanov, and the final submission is weighted average of each member's models' averages with 0.3 and 0.7 weights, i.e.:

$P_{final} = 0.3 \cdot P_{mkotyushev} + 0.7 \cdot P_{rostepifanov} = 0.3 \cdot \left( \frac{1}{2} \sum_{i=0}^{1} P_{mkotyushev, i} \right) + 0.7 \cdot \left( \frac{1}{4} \sum_{i=0}^{3} P_{rostepifanov, i} \right)$

# Reproducing the results

Pre-requisites:
- VSCode
- Docker
- NVIDIA GPU with at least 16GB of memory
- Internet connection & ~40GB of free disk space

Experiments and the final training were done using VSCode DevContainer. In the final solution, 4 models from @rostepifanov were used and two models from @mkotyushev, the following steps could be used to train the latter:

1. Clone the repository: `git clone https://github.com/mkotyushev/aorta.git`
2. Open the repository in VSCode
3. Reopen the repository in a DevContainer (`Ctrl+Shift+P` -> `Dev Containers: Reopen in Container`)
4. Make sure to activate conda environment in the terminal: `conda activate aorta`
5. Put the data into `/workspace/data` directory (i.e. `/workspace/data/images` is a directory with images, `/workspace/data/masks` is a directory with masks). To do so, first copy the data .zip archive to the `/workspace/aorta/` directory on the host machine, then move it to the `/workspace/data` directory in the DevContainer and unzip with `unzip training.zip`
6. Run the training script for folds 0 & 3: 

```
python run/main.py fit --config run/configs/common.yaml --config run/configs/unetpp.yaml --data.init_args.image_size "[128, 128, 128]" --data.init_args.batch_size 2 --trainer.accumulate_grad_batches 16 --data.init_args.fold_index 0
python run/main.py fit --config run/configs/common.yaml --config run/configs/unetpp.yaml --data.init_args.image_size "[128, 128, 128]" --data.init_args.batch_size 2 --trainer.accumulate_grad_batches 16 --data.init_args.fold_index 3
```

The expected training time is ~24 hours per fold on NVIDIA RTX 3090. 

The checkpoints are saved under `/workspace/aorta/aorta/<run id>/checkpoints/` directory, with `<run id>` being random run ID. There are two checkpoints in the directory, one for the last epoch (`last.ckpt`) and one for the best-by-`val_dice`-score-epoch (`epoch-<epoch>-step=<step>.ckpt`). The best epoch checkpoint is the one that should be used for inference.

7. Extract weights from the best saved checkpoints and put them to the `models` directory:

```
python run/ckpt_to_pt.py aorta/<fold 0 run id>/checkpoints/<fold 0 best checkpoint name>.ckpt models/<fold 0 run id>
python run/ckpt_to_pt.py aorta/<fold 3 run id>/checkpoints/<fold 3 best checkpoint name>.ckpt models/<fold 3 run id>
```

At this point, models are ready to be transferred to the inference docker container (see inference repository for further steps).
