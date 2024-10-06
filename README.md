# Introduction

This repository contains the training code for mkotyushev's part of the LOWER MATH team solution for [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/) (team's final test phase score is [TBA], securing [TBA] place). The inference code could be found in [this](https://github.com/mkotyushev/aorta_submission/) repository.

# Reproducing the results

Pre-requisites:
- VSCode
- Docker
- NVIDIA GPU with at least 16GB of memory
- Internet connection & ~40GB of free disk space

Experiments and the final training were done using VSCode DevContainer. In the final solution, 4 models from @rostepifanov were used and two models from @mkotyushev, the following steps could be used to train the latter:

1. Clone the repository: `git clone https://github.com/mkotyushev/aorta.git`
2. Open the repository in VSCode
3. Reopen the repository in a DevContainer, activate conda environment: `conda activate aorta`
4. Put the data into `/workspace/data` directory (i.e. `/workspace/data/images` is a directory with images, `/workspace/data/masks` is a directory with masks). To do so, first copy the data .zip archive to the `/workspace/aorta/` directory on host machine, then move it to the `/workspace/data` directory in the DevContainer and unzip with `unzip training.zip`
5. Run the training script for fold 0 & 3: 

```
python run/main.py fit --config run/configs/common.yaml --config run/configs/unetpp.yaml --data.init_args.image_size "[128, 128, 128]" --data.init_args.batch_size 2 --trainer.accumulate_grad_batches 16 --data.init_args.fold_index 0

python run/main.py fit --config run/configs/common.yaml --config run/configs/unetpp.yaml --data.init_args.image_size "[128, 128, 128]" --data.init_args.batch_size 2 --trainer.accumulate_grad_batches 16 --data.init_args.fold_index 3
```

The expected training time is ~24 hours per fold on NVIDIA RTX 3090. 

The checkpoints are saved under `/workspace/aorta/aorta/<run id>/checkpoints/` directory, with `<run id>` being random run ID. There are two checkpoints in the directory, one for the last epoch (`last.ckpt`) and one for the best-by-`val_dice`-score-epoch (`epoch-<epoch>-step=<step>.ckpt`). The best epoch checkpoint is the one that should be used for inference.

6. Extract weights from the best saved checkpoints and put them to the `models` directory:

```
python run/ckpt_to_pt.py aorta/<fold 0 run id>/checkpoints/<fold 0 best checkpoint name>.ckpt models/<fold 0 run id>

python run/ckpt_to_pt.py aorta/<fold 3 run id>/checkpoints/<fold 3 best checkpoint name>.ckpt models/<fold 3 run id>
```

At this point, models are ready to be transferred to the inference docker container (see inference repository for further steps).
