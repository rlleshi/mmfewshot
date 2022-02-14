import os
import os.path as osp
import mlflow

from argparse import ArgumentParser
from pathlib import Path
from rich.console import Console

CONSOLE = Console()


def parse_args():
    parser = ArgumentParser(prog='track experiments with mlflow tracking'
                                'https://mlflow.org/docs/latest/tracking.html')
    parser.add_argument(
        'experiment_name',
        help='name of experiment. Should correspond the model name')
    parser.add_argument(
        'run_name',
        help='name of experiment run. Add hyperparameters here')
    parser.add_argument('work_dir', help='dir where model files are stored')
    parser.add_argument(
        '--mlrun-dir',
        default='./mlruns',
        help='mlrun storage dir. Leave default.')
    parser.add_argument(
        '--data-dir',
        default='data/zim',
        help='path to train/meta test dataset')
    args = parser.parse_args()
    return args


def get_top_model(dir):
    return [model for model in os.listdir(dir) if model[:4] == 'best']


def find_artifact(dir, ext, hint=''):
    """Given a folder, find files based on their extension and part of name"""
    return [file for file in os.listdir(dir)
            if (osp.splitext(file)[1]==ext and hint in file)]


def main():
    args = parse_args()
    CONSOLE.print(f'Logging {args.experiment_name}-{args.run_name}...', style='green')
    Path(args.mlrun_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.mlrun_dir)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        # log artifacts
        for ext in ['.json', '.log', '.py', '.txt']:
            for artifact in find_artifact(args.work_dir, ext):
                mlflow.log_artifact(osp.join(args.work_dir, artifact))
        for artifact in find_artifact(osp.join(args.data_dir, 'meta'), '.txt'):
            mlflow.log_artifact(osp.join(args.data_dir, 'meta', artifact))

        top_model = get_top_model(args.work_dir)[0]
        if not top_model:
            CONSOLE.print(f'No best model found in {args.work_dir}')
        else:
            mlflow.log_artifact(osp.join(args.work_dir, top_model))

        # log params
        dataset_properties_path = osp.join(args.data_dir, 'meta', 'dataset.txt')
        with open(dataset_properties_path, 'r') as f:
            dataset_properties = f.read().splitlines()
        d_type = dataset_properties[0].split(' ')[1]
        sensor = dataset_properties[1].split(' ')[2]
        linewidth = dataset_properties[2].split(' ')[1]
        train_dataset = dataset_properties[4].split(' ')[1]
        filter = dataset_properties[5].split(' ')[2]

        train_acc_path = osp.join(args.work_dir, 'train_result.txt')
        with open(train_acc_path, 'r') as f:
            train_acc = f.read().splitlines()

        test_acc_path = osp.join(
            args.work_dir, find_artifact(args.work_dir, '.log', 'test')[0])
        with open(test_acc_path, 'r') as f:
            test_acc = f.read().splitlines()

        mlflow.log_params({
            'model': args.experiment_name,
            'run': args.run_name,
            'filter by record length': filter,
            'dataset type': d_type,
            'training dataset annotations': train_dataset,
            'linewidth': linewidth,
            'sensor type': sensor,
            'val acc': train_acc[0].split(' ')[1],
            'test acc mean': f'{test_acc[-2].split(" ")[-1]}%',
            'test acc std': test_acc[-1].split(" ")[-1],
        })


if __name__ == '__main__':
    main()
