import argparse
import numpy as np
from joblib import Parallel, delayed
from medpy import io
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from src.data.constants import N_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('masks_dirpath', type=Path)
    parser.add_argument('output_dirpath', type=Path)
    parser.add_argument('--n_jobs', type=int, default=10)
    return parser.parse_args()


def build_and_save_dtm(mask_filepath, output_filepath):
    mask, _ = io.load(mask_filepath)
    dtm = np.zeros((N_CLASSES, *mask.shape), dtype=np.float16)
    for i in range(N_CLASSES):
        dtm[i] = distance_transform_edt(mask == i)
    np.save(output_filepath, dtm)


def main(args):
    args.output_dirpath.mkdir(parents=True, exist_ok=True)

    mask_filepathes = sorted(list(args.masks_dirpath.glob('*.mha')))
    _ = Parallel(n_jobs=args.n_jobs)(
        delayed(build_and_save_dtm)(
            mask_filepath, 
            output_filepath=args.output_dirpath / mask_filepath.relative_to(args.masks_dirpath).with_suffix('.npy')
        ) 
        for mask_filepath in tqdm(mask_filepathes)
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
