_base_ = './retinanet_r50_fpn_2x_coco.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnest101',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        depth=101,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'
    ),
)
train_pipeline = [
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
]
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='memcached', server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf', client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
]
dist_params = dict(backend='nccl',port=25891)