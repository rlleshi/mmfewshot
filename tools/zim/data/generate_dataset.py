import glob
import subprocess
import os.path as osp
import numpy as np

from argparse import ArgumentParser
from itertools import repeat
from multiprocessing import cpu_count
from tools.zim import utils
from pathlib import Path
from rich.console import Console
from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
from functools import partial

CONSOLE = Console()
SKIP_COUNT = 0

def get_movement(sample):
    """ Get movement label given sample path"""
    return sample.split('/')[-1].split(
        '2021')[0].replace('_', ' ').lower().strip()


def generate_structure(out_dir, annotations):
    global CLASSES
    CLASSES = utils.zim_annotations_list(annotations)
    Path(osp.join(out_dir, 'meta')).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for c in CLASSES:
            Path(osp.join(out_dir, split, c)).mkdir(
                parents=True, exist_ok=True)
        open(osp.join(out_dir, 'meta', f'{split}.txt'), 'w').close()


def parse_plot_output(content):
    return content.decode('utf-8').strip('\n')


def parse_args():
    parser = ArgumentParser(prog='geneate zim dataset')
    parser.add_argument(
        'raw_dir',
        type=str,
        help='dir containing raw json files')
    parser.add_argument('--out-dir', default='data/zim/', help='output dir')
    parser.add_argument(
        '--annotations',
        default='tools/zim/data/annotations/base/valse_tango.txt',
        help='annotation text file')
    parser.add_argument(
        '--level',
        type=int,
        default=2,
        choices=[1, 2, 3],
        help='directory level of raw data')
    parser.add_argument(
        '--split',
        type=float,
        nargs='+',
        default=[0.7, 0.15, 0.15],
        help='train/val/test split')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=(cpu_count() - 2 or 1),
        help='num processes to use')
    parser.add_argument(
        '--sensor-type',
        default='acc',
        choices=['acc', 'gyro', 'both'],
        help='type of sensor to plot the data')
    parser.add_argument(
        '--axes',
        nargs='+',
        default=['x', 'y', 'z'],
        help='axes to plot')
    parser.add_argument(
        '--linewidth',
        type=int,
        default=2,
        help='width of the line')
    parser.add_argument(
        '--check-length',
        action='store_true',
        help='check bounds of length if specified')
    parser.add_argument(
        '--type',
        type=str,
        default='plot',
        choices=['plot', 'heatmap', 'clustermap', 'kde'])
    args = parser.parse_args()
    return args


def generate_image(skipped_count, items):
    sample, args = items
    script_path = 'tools/zim/plotter.py'
    np.random.seed()
    split = np.random.choice(['train', 'val', 'test'], p=args.split)
    label = '_'.join([s[0:2] for s in get_movement(sample).split(' ')])
    if label not in CLASSES:
        return

    subargs = [
        'python',
        script_path,
        sample,
        '--out-dir',
        osp.join(args.out_dir, split, label),
        '--sensor-type',
        args.sensor_type,
        '--linewidth',
        str(args.linewidth),
        '--type',
        args.type,
        '--axes',
    ]
    for ax in args.axes:
        subargs.append(ax)
    if args.check_length:
        subargs.append('--check-length')

    result = subprocess.run(subargs, capture_output=True)
    result = parse_plot_output(result.stdout)
    if result == 'sample length out of bounds':
        skipped_count[0] += 1
        return

    try:
        path = result.split("plot ")[1]
    except IndexError as e:
        CONSOLE.print(f'Error while plotting', style='red')
        CONSOLE.print(result)
        CONSOLE.print(e)
        return

    with open(osp.join(args.out_dir, 'meta', f'{split}.txt'), 'a') as ann_f:
        ann_f.write(f'{path} {LABEL_TO_NUMBER[label]}')
        ann_f.write('\n')


def main():
    args = parse_args()
    assert sum(args.split) == 1, 'train/val/test split must equal to 1'
    generate_structure(args.out_dir, args.annotations)
    global LABEL_TO_NUMBER
    LABEL_TO_NUMBER = utils.zim_label_to_number(args.annotations)
    if args.level == 1:
        data = glob.glob(osp.join(args.raw_dir, '*'))
    elif args.level == 2:
        data = glob.glob(osp.join(args.raw_dir, '*', '*'))
    elif args.level == 3:
        data = glob.glob(osp.join(args.raw_dir, '*', '*', '*'))

    CONSOLE.print(f'Found {len(data)} samples. Processing...', style='green')
    manager = Manager()
    skipped_count = manager.list([0])
    process_map(
        partial(generate_image, skipped_count),
        zip(data, repeat(args)),
        max_workers=args.num_processes,
        total=len(data))

    with open(osp.join(args.out_dir, 'meta', 'dataset.txt'), 'w') as f:
        f.write(f'Type: {args.type}\n'
                f'Sensor Type: {args.sensor_type}\n'
                f'Linewidth: {args.linewidth}\n'
                f'Train/Val/Test Split: {args.split}\n'
                f'Annotations: {args.annotations}\n'
                f'Check Length: {args.check_length}\n'
                f'Samples skipped: {skipped_count[0]}\n')


if __name__ == '__main__':
    main()
