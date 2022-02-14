import os
import os.path as osp
import numpy as np

from argparse import ArgumentParser
from rich.console import Console

from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM,
                              XGradCAM, EigenCAM, FullGrad)
from mmfewshot.classification.apis import (inference_classifier,
                                           init_classifier,
                                           process_support_images)

CONSOLE = Console()


# def parse_args():
#     parser = ArgumentParser(prog='gradcam analysis for a deep dive')
#     parser.add_argument()
#     parser.add_argument()
#     args = parser.parse_args()
#     return args


def main():
    # args = parse_args()
    checkpoint = 'mlruns/6/a5af2d2732264a0ebd1123cf82720560/artifacts/best_accuracy_mean.pth'
    config = 'mlruns/6/a5af2d2732264a0ebd1123cf82720560/artifacts/meta-baseline_conv4_1xb100_zim_8way-6shot.py'
    support_dir = osp.join('tools/zim/server/ground_truth/', 'englishvalse_8_acc_gyro', 'support_images')
    query_img = 'data/chasse_bw_bounc.jpg'

    model = init_classifier(config, checkpoint, device='cuda:0')
    files = os.listdir(support_dir)
    support_images = [osp.join(support_dir, file) for file in files]
    support_labels = [file.split('.')[0] for file in files]
    process_support_images(model, support_images, support_labels)

    CONSOLE.print(type(model))
    CONSOLE.print(model)
    target_layers = [model.backbone.layers[3].layers[2]]
    CONSOLE.print(target_layers)

    cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    pipeline = cfg.data.test.dataset.pipeline
    test_pipeline = Compose(pipeline)
    data = dict(
        img_info=dict(filename=query_img),
        gt_label=np.array(-1, dtype=np.int64),
        img_prefix=None)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    # CONSOLE.print(type(data))
    # CONSOLE.print(data)

    cam_result = cam(input_tensor=data['img'])
    CONSOLE.print(type(cam_result))

if __name__ == '__main__':
    main()
