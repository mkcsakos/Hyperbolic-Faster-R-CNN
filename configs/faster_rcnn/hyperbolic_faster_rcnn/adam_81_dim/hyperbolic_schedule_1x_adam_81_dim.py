# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=11,  # Number of epochs for a complete cycle
        eta_min=2e-7,  # Adam / AdamW
        begin=0,  # Start learning rate decay after the first epoch
        end=11,  # End after the specified epochs
        by_epoch=True,  # Step by epoch
        convert_to_iter_based=True  # Keep it epoch-based
    )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=2e-4,
        weight_decay=0.00005
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
