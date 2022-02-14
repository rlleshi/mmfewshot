# model settings
model = dict(
    type='MetaBaseline',
    backbone=dict(type='ResNet12'),
    head=dict(type='MetaBaselineHead'))


# dataset settings
img_size = 84
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

num_ways = 3 # classes for support set
num_shots = 6 # samples per each class in support set
num_queries = 4
num_val_episodes = 100
num_test_episodes = 2000

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='EpisodicDataset',
        num_episodes=100000,
        num_ways=10,
        num_shots=5,
        num_queries=5,
        dataset=dict(
            type='ZIMDataset',
            data_prefix='',
            ann_file='data/zim/meta/train.txt',
            subset='train',
            pipeline=train_pipeline)),
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
            fast_test=False,
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization is a time consuming operation
            support=dict(batch_size=num_ways * num_shots, num_workers=0),
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
            type='ZIMDataset', # dataset name
            data_prefix='',
            ann_file='data/zim/meta/test.txt',
            subset='test',
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes=num_test_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization for each task is a time consuming operation
            support=dict(batch_size=num_ways * num_shots, num_workers=0),
            query=dict(batch_size=num_ways * num_queries, num_workers=0))))


# optimizer
runner = dict(type='IterBasedRunner', max_iters=40000)
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.25,
    step=[20000, 40000])

# runtime settings
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=4000)
evaluation = dict(by_epoch=False, interval=2000, metric='accuracy')
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
pin_memory = True
use_infinite_sampler = True
seed = 0
