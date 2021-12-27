
model = dict(
    type='NanoDet',
    backbone=dict(
        type='ShuffleNetV2',
        model_size='1.0x',
        out_stages=[2, 3, 4],
        activation='LeakyReLU'),
    neck=dict(
        type='MMENanoDetPAN',
        in_channels=[116, 232, 464],
        out_channels=96,
        start_level=0,
        num_outs=3),
    bbox_head=dict(
        type='MMENanoDetHead',
        num_classes=80,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        prior_box_scale=5,
        strides=[8, 16, 32],
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss', loss_weight=0.25, reg_max=7)),

    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(type='mmdet.LoadImageFromFile', to_float32=True, file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
    dict(type='LoadAnnotations', with_bbox=True),
    # 256, 288, 320, 352, 384
    dict(
        type='mmdet.Resize',
        img_scale=[(224, 224), (480, 480)],
        multiscale_mode='range',
        keep_ratio=True),
    # dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    # dict(type='RandomAffine', max_rotate_degree=0, max_translate_ratio=0.2, scaling_ratio_range=(0.5, 1.5), max_shear_degree=0),
    # dict(type='PhotoMetricDistortion', contrast_range=(0.6, 1.4), saturation_range=(0.5, 1.2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(type='mmdet.LoadImageFromFile', to_float32=True, file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_train2017.json',
    #     img_prefix=data_root + 'images/train2017/',
    #     pipeline=train_pipeline),
    train=dict(
        _delete_=True,
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.0001,
    step=[24, 26, 27])

runner = dict(type='EpochBasedRunner', max_epochs=28)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

find_unused_parameters=True
dist_params = dict(backend='nccl', port=25987)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


work_dir = 'workdir/nanodet-m-singlescale'

