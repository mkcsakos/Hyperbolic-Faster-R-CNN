# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        eta_min=1e-4,
        begin=0,
        end=6,
        by_epoch=True
    )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(
    type='SGD',
    lr=2e-2,
    momentum=0.9,
    weight_decay=0.0005),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
