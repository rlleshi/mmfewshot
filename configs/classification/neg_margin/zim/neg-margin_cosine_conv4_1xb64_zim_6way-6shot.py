# Notes on Few Shot Classification
# https://github.com/open-mmlab/mmfewshot/blob/main/docs/intro.md
#
# Split Sets
# The classes of the dataset are divided into 3 disjoint datasets: train, test and val
# Evaluation (also called meta) set randomly samples (N way x K shot) labeled support images
# + Q unlabeled query images from the test set to form a task and get the prediction accuracy
# of query images on that task.
# Usually, the meta test repeatedly samples numerous tasks to get a sufficient evaluation and
# calculates the mean and std accuracy from all tasks.
#
#
# Pipeline
# 1) Train a model on a large dataset. This uses cross-entropy loss.
# 2) Fine-tune on few shot data. Transfer the backbone from (1) and then fine-tune a new cls head
#


# model settings
# first train head on large-scale data
# then fine-tune a new classification head on few shot data
model = dict(
    type='NegMargin',
    backbone=dict(type='Conv4'),
    head=dict(
        type='NegMarginHead',
        num_classes=10,
        in_channels=1600,
        metric_type='cosine',
        margin=-0.01,
        temperature=10.0),
    meta_test_head=dict(
        type='NegMarginHead',
        num_classes=6,
        in_channels=1600,
        metric_type='cosine',
        margin=0.0,
        temperature=5.0))


# dataset settings
img_size = 112
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

num_ways = 6 # classes
num_shots = 6 # used for training
num_queries = 4 # used for testing
num_val_episodes = 500 # ?
num_test_episodes = 2000 # ?

# config of fine-tuning using support set in Meta Test
meta_finetune_cfg = dict(
    # number of iterations in fine-tuning
    num_steps=150,
    # optimizer config in fine-tuning
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        dampening=0.9,
        weight_decay=0.001))

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=1,
    train=dict(
        # name of dataset
        type='ZIMDataset',
        data_prefix='',
        ann_file='data/zim/meta/train.txt',
        subset='train',
        pipeline=train_pipeline),
    val=dict(
        # datset wrapper for meta test also called the evaluation step
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
                num_workers=0,
                # config for fine-tuning
                train=meta_finetune_cfg),
            query=dict(
                # query set setting predict num_ways * num_queries images
                batch_size=num_ways * num_queries,
                num_workers=0))),
    test=dict(
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
                num_workers=0,
                train=meta_finetune_cfg),
            # query set predicting `batch_size` images
            query=dict(batch_size=num_ways * num_queries, num_workers=0))))


# runtime settings
runner = dict(
    # runner type and epochs for training
    type='EpochBasedRunner',
    max_epochs=200)
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001)
# most methods do not use gradient clip
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear', # warmup type
    warmup_iters=3000, # warmup iters
    warmup_ratio=0.25, # warmup ratios
    step=[15, 30, 60, 90, 120, 150]) # steps to delay the learning rate

# yapf:disable
log_config = dict(
    interval=50, # interval to print log
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
checkpoint_config = dict(interval=10)
evaluation = dict(
    by_epoch=True, # eval model by epoch
    metric='accuracy', # metric used during eval
    interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# whether to use pin-memory (speeds up memory copy operation)
pin_memory = True
use_infinite_sampler = True
seed = 0 # random seed
