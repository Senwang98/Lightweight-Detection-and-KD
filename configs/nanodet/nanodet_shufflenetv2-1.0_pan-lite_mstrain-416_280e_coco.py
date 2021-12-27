_base_ = './nanodet_shufflenetv2-1.0_pan-lite_mstrain-320_280e_coco'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', to_float32=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.Resize',
        img_scale=[(256, 256), (576, 576)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmdet.DefaultFormatBundle'),
    dict(type='mmdet.Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(416, 416),
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
    dict(type='mmdet.Resize', img_scale=(416, 416), keep_ratio=True),
    dict(type='mmdet.RandomFlip', flip_ratio=0.),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmseg.ImageToTensor', keys=['img']),
    dict(type='mmdet.Collect', keys=['img']),
]
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
