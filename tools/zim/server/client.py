import json
import socketio
from rich.console import Console

CONSOLE = Console()
sio = socketio.Client()

def get_movement(sample):
    """ Get movement label given sample path"""
    result = sample.split('/')[-1].split(
        '2021')[0].replace('_', ' ').lower().strip()
    return '_'.join([s[0:2] for s in result.split(' ')])


sample_paths = [
    # * Tango ACC
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc/Tango_2 walking steps fw_2021-11-10 10:47:52 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc/Tango_2 walking steps fw_2021-11-10 10:48:00 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc/Tango_2 walking steps fw_2021-11-10 10:48:07 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc/Tango_2 walking steps fw_2021-11-10 10:48:13 +0000.json', 'tango 2 walking steps fw'),

    # * Tango Not ACC
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc_not/Tango_2 walking steps fw_2021-11-10 10:48:21 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc_not/Tango_2 walking steps fw_2021-11-10 10:48:27 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc_not/Tango_2 walking steps fw_2021-11-10 10:48:33 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_acc_not/Tango_2 walking steps fw_2021-11-10 10:48:40 +0000.json', 'tango 2 walking steps fw'),

    # # * Tango Curved
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv/Tango_2 walking steps fw_2021-11-10 10:46:51 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv/Tango_2 walking steps fw_2021-11-10 10:46:57 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv/Tango_2 walking steps fw_2021-11-10 10:47:04 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv/Tango_2 walking steps fw_2021-11-10 10:47:10 +0000.json', 'tango 2 walking steps fw'),

    # # * Tango Not Curved
    ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv_not/Tango_2 walking steps fw_2021-11-10 10:47:17 +0000.json', 'tango 2 walking steps fw'),
    ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv_not/Tango_2 walking steps fw_2021-11-10 10:47:23 +0000.json', 'tango 2 walking steps fw'),
    ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv_not/Tango_2 walking steps fw_2021-11-10 10:47:29 +0000.json', 'tango 2 walking steps fw'),
    ('data/zim_all_version/raw_one_shot/old_first_4_batch/tango_fw_curv_not/Tango_2 walking steps fw_2021-11-10 10:47:36 +0000.json', 'tango 2 walking steps fw'),

    # * Chasse Fw Bounc Not
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc_not/English Valse_chasse fw_2021-11-10 10:49:05 +0000.json', 'valse chasse fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc_not/English Valse_chasse fw_2021-11-10 10:49:11 +0000.json', 'valse chasse fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc_not/English Valse_chasse fw_2021-11-10 10:49:18 +0000.json', 'valse chasse fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc_not/English Valse_chasse fw_2021-11-10 10:49:25 +0000.json', 'valse chasse fw'),

    # * Chasse Fw Bounc
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc/English Valse_chasse fw_2021-11-10 10:49:40 +0000.json', 'valse chasse fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc/English Valse_chasse fw_2021-11-10 10:49:46 +0000.json', 'valse chasse fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc/English Valse_chasse fw_2021-11-10 10:49:51 +0000.json', 'valse chasse fw'),
    # ('data/zim_all_version/raw_one_shot/old_first_4_batch/chasse_fw_bounc/English Valse_chasse fw_2021-11-10 10:49:57 +0000.json', 'valse chasse fw'),

    # * mixed
    # ('data/zim_all_version/raw_one_shot/raw-tango/acc and curve/Tango_2 walking steps fw_2021-12-13 10:37:40 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/raw-tango/acc and not curve/Tango_2 walking steps fw_2021-12-13 10:39:50 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/raw-tango/curve and not acc/Tango_2 walking steps fw_2021-12-13 10:41:45 +0000.json', 'tango 2 walking steps fw'),
    # ('data/zim_all_version/raw_one_shot/raw-tango/not acc not curve/Tango_2 walking steps fw_2021-12-13 10:43:11 +0000.json', 'tango 2 walking steps fw'),
]

@sio.event
def connect():
    CONSOLE.print('Connection established', style='green')
    for sample, label in sample_paths:
        content = json.load(open(sample, 'r'))
        sio.emit('inference', {'content': content, 'label': label})

@sio.event
def disconnect():
    print('disconnected from server')


if __name__ == '__main__':
    server_url = 'http://localhost:5000'
    sio.connect(server_url)
    sio.wait()