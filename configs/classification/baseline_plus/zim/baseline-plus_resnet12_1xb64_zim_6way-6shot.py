# Baseline++ performs whole classification training by replacing the top linear
# layer with a cosine classifier; afterwards the classifer is adapted to a few
# shot classification task of novel classes by performing nearest centroid or
# fine-tuning a new layer respectively

# model settings
model = dict(
    type='BaselinePlus',
    backbone=dict(type='ResNet12'),
    head=dict(
        type='CosineDistanceHead',
        num_classes=10,
        in_channels=640,
        temperature=10.0),
    meta_test_head=dict(
        type='CosineDistanceHead',
        num_classes=6,
        in_channels=640,
        temperature=5.0))

# dataset settings
img_size = 140
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(int(img_size * 1.15), -1)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

num_ways = 6
num_shots = 6
num_queries = 4
num_val_episodes = 300
num_test_episodes = 2000

meta_finetune_cfg = dict(
    num_steps=600,
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001))

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='ZIMDataset',
        ann_file='data/zim/meta/train.txt',
        data_prefix='',
        subset='train',
        pipeline=train_pipeline),
    val=dict(
        type='MetaTestDataset',
        num_episodes=num_val_episodes,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        dataset=dict(
            type='ZIMDataset',
            data_prefix='',
            ann_file='data/zim/meta/val.txt',
            subset='val',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=num_val_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization is a time consuming operation
            support=dict(
                batch_size=4, drop_last=True, train=meta_finetune_cfg),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))),
    test=dict(
        type='MetaTestDataset',
        num_episodes=num_test_episodes,
        num_ways=num_ways,
        num_shots=num_shots,
        num_queries=num_queries,
        # seed for generating meta test episodes
        episodes_seed=0,
        dataset=dict(
            type='ZIMDataset',
            subset='test',
            data_prefix='',
            ann_file='data/zim/meta/test.txt',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=num_test_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization for each task is a time consuming operation
            support=dict(
                batch_size=4, drop_last=True, train=meta_finetune_cfg),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))))

# runner settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.25,
    step=[50, 80, 120, 160])

# runtime settings
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=10)
evaluation = dict(by_epoch=True, metric='accuracy', interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
pin_memory = True
use_infinite_sampler = True
seed = 0
