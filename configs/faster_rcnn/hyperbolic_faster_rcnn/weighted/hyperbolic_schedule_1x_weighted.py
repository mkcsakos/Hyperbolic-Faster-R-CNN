# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',
        monitor='coco/bbox_mAP_50',
        rule='greater',
        factor=0.8,
        patience=1,
        by_epoch=True,
        verbose=True
    ),

    dict(
        type='CosineAnnealingLR',
        T_max=7330 * 11,  # Adjust the number of iterations for one cycle
        eta_min=0.000005,  # Adam / AdamW
        # eta_min=0.0001, # SGD
        begin=500,
        end=7330 * 12,
        by_epoch=False,
        convert_to_iter_based=True
    )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=0.0005,
        weight_decay=0.00005
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
