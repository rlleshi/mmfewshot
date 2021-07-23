_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py', '../_base_/schedules/schedule.py',
    '../_base_/default_runtime.py'
]

model = dict(type='TestDetection')
data = dict(samples_per_gpu=1)
