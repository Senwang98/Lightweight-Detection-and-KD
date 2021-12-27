_base_ = [
#     '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.001
beta_fgd = 0.0005
gamma_fgd = 0.0005
lambda_fgd = 0.000005
distiller = dict(
    type='CSD_DetectionDistiller',
    teacher_pretrained='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth',
    init_student=True,
    distill_cfg=[dict(student_module='neck.fpn_convs.4.conv',
                       teacher_module='neck.fpn_convs.4.conv',
                       output_hook=True,
                       methods=[dict(type='CSDLoss',
                                     name='loss_csd_fpn_4',
                                     student_channels=256,
                                     teacher_channels=256,
                                     temp=temp,
                                     alpha_fgd=alpha_fgd,
                                     beta_fgd=beta_fgd,
                                     gamma_fgd=gamma_fgd,
                                     lambda_fgd=lambda_fgd,
                                     )
                                ]
                      ),
                 dict(student_module='neck.fpn_convs.3.conv',
                      teacher_module='neck.fpn_convs.3.conv',
                      output_hook=True,
                      methods=[dict(type='CSDLoss',
                                     name='loss_csd_fpn_3',
                                     student_channels=256,
                                     teacher_channels=256,
                                     temp=temp,
                                     alpha_fgd=alpha_fgd,
                                     beta_fgd=beta_fgd,
                                     gamma_fgd=gamma_fgd,
                                     lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.2.conv',
                      teacher_module='neck.fpn_convs.2.conv',
                      output_hook=True,
                      methods=[dict(type='CSDLoss',
                                     name='loss_csd_fpn_2',
                                     student_channels=256,
                                     teacher_channels=256,
                                     temp=temp,
                                     alpha_fgd=alpha_fgd,
                                     beta_fgd=beta_fgd,
                                     gamma_fgd=gamma_fgd,
                                     lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.1.conv',
                      teacher_module='neck.fpn_convs.1.conv',
                      output_hook=True,
                      methods=[dict(type='CSDLoss',
                                     name='loss_csd_fpn_1',
                                     student_channels=256,
                                     teacher_channels=256,
                                     temp=temp,
                                     alpha_fgd=alpha_fgd,
                                     beta_fgd=beta_fgd,
                                     gamma_fgd=gamma_fgd,
                                     lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),
                 dict(student_module='neck.fpn_convs.0.conv',
                      teacher_module='neck.fpn_convs.0.conv',
                      output_hook=True,
                      methods=[dict(type='CSDLoss',
                                     name='loss_csd_fpn_0',
                                     student_channels=256,
                                     teacher_channels=256,
                                     temp=temp,
                                     alpha_fgd=alpha_fgd,
                                     beta_fgd=beta_fgd,
                                     gamma_fgd=gamma_fgd,
                                     lambda_fgd=lambda_fgd,
                                    )
                               ]
                      ),

                 ]
)

student_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_r101_fpn_2x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
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
evaluation = dict(interval=1, metric='bbox')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
]