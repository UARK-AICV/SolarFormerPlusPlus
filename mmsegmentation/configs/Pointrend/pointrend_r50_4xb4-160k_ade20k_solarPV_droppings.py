_base_ = [
    '../_base_/models/pointrend_r50.py',
    '../_base_/datasets/solarPV_Droppings.py','../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=[
        dict(
            type='FPNHead',
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=-1,
            num_classes=3,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='PointHead',
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])

param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=200),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=200,
        end=160000,
        by_epoch=False,
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100000, val_interval=1000)

checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, save_best='mIoU', max_keep_ckpts=1),

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, save_best='mIoU', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
