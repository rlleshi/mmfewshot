# model settings
model = dict(
    type='MetaBaseline',
    backbone=dict(type='Conv4'),
    head=dict(type='MetaBaselineHead'))

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

num_ways = 2
num_shots = 6
num_queries = 4
num_val_episodes = 100
num_test_episodes = 2000
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='EpisodicDataset',
        num_episodes=50000,
        num_ways=6,
        num_shots=5,
        num_queries=5,
        dataset=dict(
            type='ZIMDataset',
            data_prefix='',
            ann_file='data/zim/meta/train.txt',
            subset='train',
            pipeline=train_pipeline)),
    val=dict(
        # datset wrapper for meta test also called the evaluation step
        # this is where the few-shot fine-tuning happens
        type='MetaTestDataset',
        # total number of test tasks
        num_episodes=num_val_episodes,
        # number of class in each task
        num_ways=num_ways,
        # number of support images in each task
        num_shots=num_shots,
        # number of query images in each task
        num_queries=num_queries,
        dataset=dict(
            type='ZIMDataset',
            data_prefix='',
            ann_file='data/zim/meta/val.txt',
            subset='val',
            pipeline=test_pipeline),
        meta_test_cfg=dict( # config of meta testing
            num_episodes=num_val_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods
            # for testing acceleration
            fast_test=True,
            # dataloader setting for feature extraction of fast test
            test_set=dict(batch_size=16, num_workers=2),
            # worker initialization is a time consuming operation
            # support set setting in meta test
            support=dict(
                # batch size for fine-tuning
                batch_size=num_ways * num_shots,
                # can set to 0 for few images
                num_workers=0),
            query=dict(
                # query set setting predict num_ways * num_queries images
                batch_size=num_ways * num_queries,
                num_workers=0))),
    test=dict(
        # in the test phase we also use different classes and we also
        # perform meta testing
        type='MetaTestDataset',
        # total number of test tasks
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
            # total number of test tasks
            num_episodes=num_test_episodes,
            num_ways=num_ways,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            # support set setting in meta test
            support=dict(
                batch_size=num_ways * num_shots,
                num_workers=0),
            # query set predicting `batch_size` images
            query=dict(batch_size=num_ways * num_queries, num_workers=0))))

runner = dict(type='IterBasedRunner', max_iters=50000)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.25,
    step=[20000, 40000])
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
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
