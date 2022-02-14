import subprocess
import schedule
from rich.console import Console

CONSOLE = Console()
TRAIN_SCRIPT = 'tools/classification/dist_train.sh'

def train_meta_baseline():
    CONSOLE.print('Starting to train meta baseline...', style='green')
    config = 'configs/classification/meta_baseline/zim/meta-baseline_conv4_1xb100_zim_4way-5shot.py'
    n_gpu = str(1)
    work_dir = 'work_dir/zim_meta/'

    subargs = [
        'bash',
        TRAIN_SCRIPT,
        config,
        n_gpu,
        '--work-dir',
        work_dir
    ]
    subprocess.run(subargs)
    return schedule.CancelJob


def train_relation_net():
    CONSOLE.print('Starting to train relation-net...', style='green')
    config = 'configs/classification/relation_net/zim/relation-net_conv4_1xb105_zim_5way-5shot.py'
    n_gpu = str(1)
    work_dir = 'work_dir/zim_relation_net/'

    subargs = [
        'bash',
        TRAIN_SCRIPT,
        config,
        n_gpu,
        '--work-dir',
        work_dir
    ]
    subprocess.run(subargs)
    return schedule.CancelJob


def train_proto_net():
    CONSOLE.print('Starting to train proto-net...', style='green')
    config = 'configs/classification/proto_net/zim/proto-net_conv4_1xb105_cub_5way-5shot.py'
    n_gpu = str(1)
    work_dir = 'work_dir/zim_proto_net/'

    subargs = [
        'bash',
        TRAIN_SCRIPT,
        config,
        n_gpu,
        '--work-dir',
        work_dir
    ]
    subprocess.run(subargs)
    return schedule.CancelJob


schedule.every().tuesday.at('04:00').do(train_meta_baseline)
# schedule.every().saturday.at('03:00').do(train_relation_net)
# schedule.every().sunday.at('12:00').do(train_proto_net)


while True:
    schedule.run_pending()
