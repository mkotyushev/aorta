import argparse
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cpkt_filepaths', type=Path, nargs='+', help='Path(s) to checkpoint files')
    parser.add_argument('output_dirpath', type=Path, help='Path to the directory to save the pytorch model files')
    return parser.parse_args()


def main(args):
    args.output_dirpath.mkdir(parents=True, exist_ok=True)
    for cpkt_filepath in args.cpkt_filepaths:
        output_filepath = args.output_dirpath / cpkt_filepath.with_suffix('.pt').name
        print(f'Converting {cpkt_filepath} to {output_filepath}')
        ckpt = torch.load(cpkt_filepath)
        state_dict = ckpt['state_dict']
        torch.save(state_dict, output_filepath)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
