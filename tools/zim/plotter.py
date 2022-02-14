import json
import string
import glob

import pandas as pd
import seaborn as sns
import os.path as osp
import numpy as np
import random as rd
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from rich.console import Console
from pathlib import Path
from PIL import Image
from itertools import combinations

CONSOLE = Console()

LENGTH_BOUNDS = {
    'en_va_ch_bg': [78, 123],
    'en_va_ch_fw': [75, 141],
    'en_va_na_tu_bw_1_-_3': [67, 129],
    'en_va_na_tu_fw_1_-_3': [63, 132],
    'en_va_na_tu_st_bw_1_-_6': [130, 186],
    'en_va_na_tu_st_fw_1_-_6': [80, 186],
    'ta_2_wa_st_bw': [46, 95],
    'ta_2_wa_st_fw': [50, 95],
    'ta_ro_tu_st_bw': [160, 229],
    'ta_ro_tu_st_fw': [163, 237],
}


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def get_movement_1(sample):
    """ Get movement label given sample path based on the format type
        the format type depends on the frontend"""

    year = '2021'
    result = sample.split('/')[-1].split(
        year)[0].replace('_', ' ').lower().strip()

    if result.endswith('.json'):
        year = '2022'
        result = sample.split('/')[-1].split(
        year)[0].replace('_', ' ').lower().strip()

    return '_'.join([s[0:2] for s in result.split(' ')])


def crop_white(im_path):
    img = Image.open(im_path)
    img_np = np.array(img)
    white = np.array([255, 255, 255])
    mask = np.abs(img_np - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    result = img_np[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    plt.imsave(im_path, result)


def parse_args():
    parser = ArgumentParser(
        prog='different kinds of plots for accelero and gyro sensor data')
    parser.add_argument('input', help='path to sample or directory of samples')
    parser.add_argument(
        '--sensor-type',
        type=str,
        default='acc',
        choices=['both', 'acc', 'gyro'],
        help='type of sensor to plot')
    parser.add_argument(
        # TODO: currectly only works on heat- and cluster- map
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


def make_plot(args, content, sample):
    if args.sensor_type == 'both':
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
        balancer = 0
    elif args.sensor_type == 'acc':
        sensors = ['acc_x', 'acc_y', 'acc_z']
        balancer = 0
    else:
        sensors = ['gy_x', 'gy_y', 'gy_z']
        balancer = 3

    ind_to_sensor = {i+balancer: sensor for i, sensor in enumerate(sensors)}
    palette = ['Reds', 'Blues', 'Greys', 'Oranges', 'Purples', 'Greens']
    results = {k: [] for k in sensors}

    for row in content:
        for i in range(len(sensors)):
            results[ind_to_sensor[i+balancer]].append(row[i+balancer])

    if args.check_length:
        if sample is None:
            CONSOLE.print('check length only works with raw json files',
                          style='yellow')
        else:
            lower_b, upper_b = LENGTH_BOUNDS[get_movement_1(sample)]
            length = len(results[list(results.keys())[0]])
            if (length < lower_b) or (length > upper_b):
                CONSOLE.print('sample length out of bounds', style='yellow')
                return

    sns.set(rc={'figure.figsize': (15, 13)})
    out_dir = osp.join(args.out_dir, f'{gen_id()}.{args.img_ext}')

    if args.type == 'plot':
        for i, k in enumerate(results.keys()):
            df = pd.DataFrame({'Sample Index': list(range(0, len(results[k]))),
                            'Sensor Reading': results[k], 'Sensor': k})
            fig = sns.lineplot(x='Sample Index', y='Sensor Reading',
                            data=df, hue='Sensor', palette=palette[i],
                            legend=None, linewidth=args.linewidth)
            fig.axis('off')

        output = fig.get_figure()
        output.savefig(out_dir, bbox_inches='tight')
        output.clf() # clear the figure
    elif args.type == 'kde':
        for pair in combinations(list(results.keys()), r=2):
            # x-y; x-z; y-z; in a single kde plot
            fig = sns.jointplot(
                x=results[pair[0]],
                y=results[pair[1]], kind='kde')

        fig.ax_marg_x.set_axis_off()
        fig.ax_marg_y.set_axis_off()
        fig.ax_joint.set_axis_off()

        plt.savefig(out_dir, dpi=500, bbox_inches='tight', pad_inches=0)

    CONSOLE.print(f'Saved plot {out_dir}', style='green')


def make_heatmap(args, content, sample):
    out_dir = osp.join(args.out_dir, f'{gen_id()}.{args.img_ext}')
    results = []
    axes_to_index = {
        'x': 0,
        'y': 1,
        'z': 2,
    }

    for row in content:
        if args.sensor_type == 'acc' or args.sensor_type == 'both':
            offset = 0
        elif args.sensor_type == 'gyro':
            offset = 3

        temp = []
        for ax in args.axes:
            temp.append(row[axes_to_index[ax]+offset])

        if args.sensor_type == 'both':
            offset = 3
            for ax in args.axes:
                temp.append(row[axes_to_index[ax]+offset])

        results.append(temp)

    if args.check_length:
        if sample is None:
            CONSOLE.print('check length only works with raw json files',
                          style='yellow')
        else:
            lower_b, upper_b = LENGTH_BOUNDS[get_movement_1(sample)]
            length = len(results)
            if (length < lower_b) or (length > upper_b):
                CONSOLE.print('sample length out of bounds', style='yellow')
                return

    results = np.array(results)
    results = np.swapaxes(results, 0, 1)

    if args.type == 'clustermap':
        ax = sns.clustermap(results, linewidth=0, cbar=False)
        ax.ax_row_dendrogram.set_visible(False)
        ax.ax_col_dendrogram.set_visible(False)
        ax.ax_heatmap.tick_params(tick2On=False, labelsize=False)
    else:
        _ = sns.heatmap(results, linewidth=0, cbar=False)

    plt.axis('off')
    plt.savefig(out_dir, dpi=500, bbox_inches='tight',
                transparent=True, pad_inches=0)
    CONSOLE.print(f'Saved plot {out_dir}', style='green')

    if args.type == 'clustermap':
        crop_white(out_dir)


def main(args, content, sample=None):
    if args.swap_axes:
        # ! This idea probably does not work
        new_content = []
        radian_thr = 1.5708

        for row in content:
            temp = []
            # gyro-x is greater than 90 degrees either forward or backward
            CONSOLE.print(row[3])
            if abs(row[3]) >= radian_thr:
                CONSOLE.print('Swapping axes...', style='red')
                temp.append(row[0]) # x
                temp.append(row[2]) # z
                temp.append(row[1]) # y
            else:
                temp.append(row[0]) # x
                temp.append(row[1]) # y
                temp.append(row[2]) # z

            for i in range(3, 6):
                temp.append(row[i])
            new_content.append(temp)

        content = new_content

    make_output = make_plot if args.type in ['plot', 'kde'] else make_heatmap
    make_output(args, content, sample)


if __name__ == '__main__':
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if osp.isdir(args.input):
        input = [inp for inp in glob.glob(osp.join(args.input, '*')) if inp.endswith('.json')]
    else:
        input = [args.input]

    for sample in input:
        content = json.load(open(sample, 'r'))
        main(args, content, sample)



### server code for swapping axes
# radian_thr_p = 1.5708
# radian_thr_n = -1.5708
# axes_map = {'x': 0, 'y': 1, 'z': 2}
# axes_layout_default = ['x', 'y', 'z']
# axes_layout_swapped = ['x', 'z', 'y']

# result_content = []

# # first 10 values to find out orientation
# orientation_sum = 0
# for i in range(10):
#     CONSOLE.print(content[i][3])
#     orientation_sum += content[i][3]

# if orientation_sum > 0:
#     CONSOLE.print('Setting initial orientation: Down - Up', style='yellow')
#     axes_layout = axes_layout_swapped
# else:
#     CONSOLE.print('Setting initial orientation: Up - Down', style='yellow')
#     axes_layout = axes_layout_default

# for row in content:
#     temp = []
#     CONSOLE.print(row[3])
#     if row[3] > radian_thr_p and axes_layout == axes_layout_swapped:
#         axes_layout = axes_layout_default
#         CONSOLE.print('Swapping axes to default x-y-z', style = 'red')
#     elif row[3] < radian_thr_n and axes_layout == axes_layout_default:
#         CONSOLE.print('Swapping axes to x-z-y', style='red')

#     temp.append(row[axes_map[axes_layout[0]]])
#     temp.append(row[axes_map[axes_layout[1]]])
#     temp.append(row[axes_map[axes_layout[2]]])

#     # add gyro
#     temp.append(row[3])
#     temp.append(row[4])
#     temp.append(row[5])

#     result_content.append(temp)
# ###
