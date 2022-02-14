import glob
import os
import os.path as osp

from argparse import ArgumentParser
from rich.console import Console
from tools.zim import utils


CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(prog='geneate file list for one-shot classes')
    parser.add_argument('annotations', help='annotations')
    parser.add_argument(
        '--in-dir',
        default='data/zim/val',
        help='directory of split with classes')
    parser.add_argument(
        '--out-dir',
        default='data/zim/',
        help='directory of classes with samples')
    parser.add_argument(
        '--level',
        type=int,
        default=2,
        choices=[1, 2, 3],
        help='directory level of raw data')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    LABEL_TO_NUMBER = utils.label_to_number(args.annotations)
    out = osp.join(
        args.out_dir, 'meta', f'{args.in_dir.split(os.sep)[-1]}.txt')

    if args.level == 1:
        data = glob.glob(osp.join(args.in_dir, '*'))
    elif args.level == 2:
        data = glob.glob(osp.join(args.in_dir, '*', '*'))
    elif args.level == 3:
        data = glob.glob(osp.join(args.in_dir, '*', '*', '*'))

    with open(out, 'a') as ann_f:
        for path in data:
            ann_f.write(f'{path} {LABEL_TO_NUMBER[path.split(os.sep)[-2]]}')
            ann_f.write('\n')

if __name__ == '__main__':
    main()
