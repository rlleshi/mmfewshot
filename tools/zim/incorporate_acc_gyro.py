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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if osp.isdir(args.input):
        inputs = [inp for inp in glob.glob(osp.join(args.input, '*')) if inp.endswith('.json')]
    else:
        inputs = [args.input]

    radian_thr = 1.5708 # no sample is longer than this

    for sample in inputs:
        result = []
        content = json.load(open(sample, 'r'))

        for row in content:
            temp = []
            if abs(row[3]) > radian_thr:
                temp.append(row[0])
                temp.append(row[2])
                temp.append(row[1])
            else:
                temp.append(row[0])
                temp.append(row[1])
                temp.append(row[2])

            result.append(temp)

        plotter.main(args, result)

if __name__ == '__main__':
    main()
