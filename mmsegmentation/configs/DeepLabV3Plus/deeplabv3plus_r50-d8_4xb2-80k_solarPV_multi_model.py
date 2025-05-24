_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/solarPV_Multi_Model.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, 
    decode_head=dict(
        num_classes=7,
    ),
    auxiliary_head=dict(
        num_classes=7,
    ),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(50, 50)),
    )

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
