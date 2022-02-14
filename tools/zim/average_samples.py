import os
import json
import glob
import numpy as np
import os.path as osp

from argparse import ArgumentParser
from pathlib import Path
from rich.console import Console
from tools.zim import plotter

CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(prog='average plots')
    parser.add_argument('input', help='path to directory of samples')
    parser.add_argument(
        '--sensor-type',
        type=str,
        default='acc',
        choices=['both', 'acc', 'gyro'],
        help='type of sensor to plot')
    parser.add_argument(
        '--axes',
        nargs='+',
        default=['x', 'y', 'z'],
        help='axes to plot')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='data/',
        help='out dir to save results')
    parser.add_argument(
        '--linewidth',
        type=int,
        default=2,
        help='line width for plot')
    parser.add_argument(
        '--check-length',
        action='store_true',
        help='check bounds of length if specified')
    parser.add_argument(
        '--img-ext',
        default='jpg',
        choices=['jpg', 'png'],
        help='out image extension')
    parser.add_argument(
        '--type',
        type=str,
        default='plot',
        choices=['plot', 'heatmap', 'clustermap', 'kde'],
        help='type of plot to produce')
    parser.add_argument(
        '--swap-axes',
        action='store_true',
        help='swap y & z axes based on x-gyro')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert osp.isdir(args.input), 'provide the path to the input directory'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    inputs = [inp for inp in glob.glob(osp.join(args.input, '*')) if inp.endswith('.json')]

    max_length = 250 # no sample is longer than this
    min_length = 40
    num_readings = 6
    result = np.zeros(shape=(max_length, num_readings))
    lengths = {} # {length: #samples_greater_or_equal_than}

    for sample in inputs:
        content = json.load(open(sample, 'r'))
        length = len(content)
        if length < min_length:
            CONSOLE.print('Sample too short, skipping', style='yellow')
            continue

        # CONSOLE.print(f'Length: {length}', style='green')
        if lengths.get(length, None) is None:
            lengths[length] = 0
            mIn = max_length
            for l in lengths.keys():
                if length < l and l < mIn:
                    mIn = l
                    lengths[length] = lengths[l]

        for l in lengths.keys():
            if length >= l:
                lengths[l] += 1

        for i in range(length):
            result[i] += content[i]

    # remove zero entries: get the indexes of entiries that are zero
    # bitwise flip them with tilde and then use the indexes of all non-zero
    # elements to pick the correct values
    result = result[~np.all(result==0, axis=1)]

    # divide by how many samples there are for each length
    # CONSOLE.print(lengths, style='yellow')
    lengths = {k: v for k, v in sorted(list(lengths.items()))}
    prev_l = None

    for l in lengths.keys():
        # CONSOLE.print(f'{prev_l} - {l}')
        if prev_l is not None:
            result[prev_l:l] /= lengths[l]
        else:
            result[:l] /= lengths[l]
        prev_l = l

    # for i in range(len(result)):
    #     if i in list(lengths.keys()):
    #         CONSOLE.print('\n ===== \n', style='yellow')
    #     CONSOLE.print(result[i])
    plotter.main(args, content)

if __name__ == '__main__':
    main()
