import time
import os.path as osp
import os
import eventlet
import socketio

from rich.console import Console
from argparse import ArgumentParser
from mmfewshot.classification.apis import (inference_classifier,
                                           init_classifier,
                                           process_support_images)
from tools.zim import plotter

CONSOLE = Console()

# https://stackoverflow.com/questions/65436640/swift-socket-io-compatible-version-with-server-python-socketio
# D_TYPE = 'heatmap_groupped_simplified'
# PATH = 'mlruns/5/bc7465300dec45bdb7ad5a46c8515794/artifacts/' # 4way

# D_TYPE = 'heatmap_groupped'
# PATH = 'mlruns/1/70ae18a3c7bb453ca2c024de3c856124/artifacts/' # 5way

# D_TYPE = 'heatmap_acc_gyro'
# PATH = 'mlruns/1/b513b16cfc2c4bae9c4b21c850288e3c/artifacts/' # 5way

# D_TYPE = 'englishvalse_8_acc_gyro'
# PATH = 'mlruns/6/a5af2d2732264a0ebd1123cf82720560/artifacts/' # 8way

MODEL = {
    '2 walking steps fw': None,
    'chasse fw': None,
    'chasse bg': None,
    'natural turn bw 1 - 3': None,
    'natural turn fw 1 - 3': None
}

DEVICE = 'cuda:0'
ARTIFACTS = 'artifacts'
CHECKPOINT = 'best_accuracy_mean.pth'

PATH = {
    '2 walking steps fw': 'mlruns/6/00e2e82311334949a0d22c73b276c111',
    'chasse fw': 'mlruns/6/873bfc339ea047f9a30ac58d82f627af',
    'chasse bg': 'mlruns/6/3618400f91f946cca6a7b2138a455e07',
    'natural turn bw 1 - 3': 'mlruns/6/102db7575c2a4591a2152ac4c3e6f12c',
    'natural turn fw 1 - 3': 'mlruns/6/e3d323a7876946ca8e77d019ac7cc871'
}

# TODO: 2x models; 3x; 5x; 7x; 10x
D_TYPE = {
    # '2 walking steps fw': {'2way/tango2w_f2': [0, 1]}, first try just to add more samples

    '2 walking steps fw': '2way/tango2w_f2',
    'chasse fw': '2way/chasse_fw',
    'chasse bg': '2way/chasse_bw',
    'natural turn bw 1 - 3': '2way/nt_bw_1_3',
    'natural turn fw 1 - 3': '2way/nt_fw_1_3',
}

two_way_6_shot = 'meta-baseline_conv4_1xb100_zim_2way-6shot.py'
CONFIG = {
    '2 walking steps fw': osp.join(PATH['2 walking steps fw'], ARTIFACTS, 'meta-baseline_conv4_1xb100_zim_4way-6shot.py'),
    'chasse fw': osp.join(PATH['chasse fw'], ARTIFACTS, two_way_6_shot),
    'chasse bg': osp.join(PATH['chasse bg'], ARTIFACTS, two_way_6_shot),
    'natural turn bw 1 - 3': osp.join(PATH['natural turn bw 1 - 3'], ARTIFACTS, two_way_6_shot),
    'natural turn fw 1 - 3': osp.join(PATH['natural turn fw 1 - 3'], ARTIFACTS, two_way_6_shot),
}

length_bounds = {
    'chasse fw': [77, 140],
    'chasse bg': [78, 123],
    '2 walking steps fw': [45, 103],
    'natural turn bw 1 - 3': [67, 129],
    'natural turn fw 1 - 3': [63, 132],
}

filter_map = {
    'chasse fw': ['chasse_fw_bounc_not', 'chasse_fw_bounc'],
    '2 walking steps fw': ['acc_curve', 'acc_not_curve_not', 'acc_curve_not', 'acc_not_curve'],
    'chasse bg': ['chasse_bw_bounc_not', 'chasse_bw_bounc'],
    'natural turn bw 1 - 3': ['nt_bw_1-3_rising', 'nt_bw_1-3_rising_not'],
    'natural turn fw 1 - 3': ['nt_fw_1-3_rising', 'nt_fw_1-3_rising_not'],
}

sio = socketio.Server(
    # logger=True,
    # engineio_logger=True,
    cors_allowed_origins='*')

app = socketio.Middleware(sio)
# app = socketio.WSGIApp(sio, static_files={
#     '/': {'content_type': 'text/html', 'filename': 'index.html'}
# })


# @sio.event
@sio.on('connect')
def connect(sid, environ=None):
    """ Invoked everytime a client connects.

        sid: client id
        environ: dict in standard WSGI format containing request informations
                 and HTTP headers
        auth: contains any authentication details that might have been passed
              by the client"""

    global MODEL
    if not all(MODEL.values()):
        CONSOLE.print(f'Client {sid} connected', style='bold green')
    else:
        CONSOLE.print(f'Client {sid} re-connected', style='green')
        return

    CONSOLE.print('Bootstraping models...', style='green')
    st = time.time()
    for key in MODEL.keys():
        if MODEL.get(key, None) is not None:
            continue
        MODEL[key] = init_classifier(CONFIG[key], osp.join(PATH[key], ARTIFACTS, CHECKPOINT), device=DEVICE)

        # prepare support set, each support class only contains one shot
        support_dir = osp.join('tools/zim/server/ground_truth/', D_TYPE[key], 'support_images')
        files = os.listdir(support_dir)
        support_images = [osp.join(support_dir, file) for file in files]
        support_labels = [file.split('.')[0] for file in files]
        process_support_images(MODEL[key], support_images, support_labels)

    CONSOLE.print(f'Finished bootstrap. Took {time.time() - st}', style='green')


# @sio.event
@sio.on('inference')
def inference(sid, data):
    """Perform inference given data

    Args:
        sid ([type]): [description]
        data ([dict])): [dict containing content of data]
    """
    content = data['content']
    label = data['label']
    if MODEL.get(label, 'not') is None:
        sio.emit('inference_result', 'Model not boostraped. Emit the `connect` event to initialize it first.')
        return
    elif MODEL.get(label, 'not') == 'not':
        sio.emit('inference_result', 'This movement does not have evaluations.')
        return

    lower_b, upper_b = length_bounds[label]
    content_length = len(content)
    if (content_length < lower_b):
        msg = 'Movement was too fast. Please try again.'
        CONSOLE.print(msg, style='yellow')
        sio.emit('inference_result', msg)
        return
    elif (content_length > upper_b):
        msg = 'Movement was too slow. Please try again.'
        CONSOLE.print(msg, style='yellow')
        sio.emit('inference_result', msg)
        return

    query_dir = osp.join('tools/zim/server/ground_truth/', D_TYPE[label], 'query_images')
    parser = ArgumentParser()
    args = parser.parse_args()
    args.sensor_type = 'both'
    args.out_dir = query_dir
    args.check_length = False
    args.img_ext = 'jpg'
    args.type = 'heatmap'
    args.linewidth = 8
    args.axes = ['x', 'y', 'z']
    args.swap_axes = False

    plotter.main(args, content)

    query_image = osp.join(query_dir, os.listdir(query_dir)[0])
    result = inference_classifier(MODEL[label], query_image)

    # os.remove(query_image)
    CONSOLE.print(result)

    # TODO: filtering is now redundant
    filter = filter_map.get(label, None)
    if filter is None:
        prediction = 'This movement does not have evaluations'
    else:
        result = {k: v for k, v in result.items() if k in filter}
        prediction = max(result, key=result.get) if result else ''

    CONSOLE.print(f'Final prediction: {prediction}', style='bold yellow')
    sio.emit('inference_result', prediction)


# @sio.event
@sio.on
def disconnect(sid):
    CONSOLE.print(f'disconnected {sid}', style='yellow')


if __name__ == '__main__':
    # * python-engineio=2.2.0; python-socketio=2.0.0
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
