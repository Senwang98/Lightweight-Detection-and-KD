# _base_ = './nanodet_shufflenetv2-1.0_pan-lite_mstrain-320_280e_coco'
_base_ = './nanodet_shufflenetv2-1.0_pan-lite-da-first-320_280e_coco.py'

model = dict(
    backbone=dict(widen_factor=1.5),
    neck=dict(in_channels=[176, 352, 704], out_channels=128),
    bbox_head=dict(in_channels=128, feat_channels=128))

dist_params = dict(backend='nccl', port=25999)
