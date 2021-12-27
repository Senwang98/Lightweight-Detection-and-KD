# model settings
model = dict(
    type='mmdet.SingleStageDetector',
    backbone=dict(
        type='nanodet.mmcls_ShuffleNetV2',
        widen_factor=1.0,
        out_indices=(0, 1, 2),
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(type='Pretrained', checkpoint='mmcls://shufflenet_v2')),
    neck=dict(
        type='nanodet.NanoDetPAN',
        in_channels=[116, 232, 464],
        out_channels=96,
        start_level=0,
        num_outs=3),
    bbox_head=dict(
        type='nanodet.NanoDetHead',
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
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='mmdet.ATSSAssigner', topk=9),
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
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='mmdet.LoadImageFromFile', to_float32=True),
    dict(type='mmdet.LoadImageFromFile', to_float32=True, file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    # dict(
    #     type='mmdet.Resize',
    #     img_scale=[(224, 224), (480, 480)],
    #     multiscale_mode='range',
    #     keep_ratio=True),
    dict(
        type='RandomAffine',
        max_rotate_degree=0,
        max_translate_ratio=0.2,
        scaling_ratio_range=(0.5, 1.5),
        max_shear_degree=0),
    dict(
        type='PhotoMetricDistortion',
        contrast_range=(0.6, 1.4),
        saturation_range=(0.5, 1.2)),
    dict(type='mmdet.Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmdet.DefaultFormatBundle'),
    dict(type='mmdet.Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    # dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadImageFromFile', file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.DefaultFormatBundle'),
            dict(type='mmdet.Collect', keys=['img'])
        ])
]
demo_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='mmdet.RandomFlip', flip_ratio=0.),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmseg.ImageToTensor', keys=['img']),
    dict(type='mmdet.Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=96,
    workers_per_gpu=4,
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
test_setting = dict(
    repo='mmdet', single_gpu_test=dict(show=False), multi_gpu_test=dict())
evaluation = dict(
    type='mme.EvalHook',
    dataset=data['val'],
    dataloader=dict(samples_per_gpu=2, workers_per_gpu=2),
    test_setting=test_setting,
    by_epoch=True,
    interval=1,
    metric='bbox')
demo_setting = dict(
    demo_pipeline=demo_pipeline,
    forward_kwargs=dict(return_loss=False, rescale=True),
    formatter=dict(type='mme.DetResultFormatter', extract_list=True))

# optimizer
optimizer = dict(type='SGD', lr=0.14*2, momentum=0.9, weight_decay=1.0e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1,
    step=[24, 26, 27])
runner = dict(type='EpochBasedRunner', max_epochs=28)

# runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl', port=25966)
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
