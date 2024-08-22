# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# warmup_end_step = 1000

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=12,
    #     by_epoch=True,
    #     milestones=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     gamma=0.8
    # ),
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
        # T_max=100, # TEST
        eta_min=0.000005,  # Adam / AdamW
        # eta_min=0.0001, # SGD
        begin=500,
        end=7330 * 12,
        # end=100, # TEST
        by_epoch=False,
        # convert_to_iter_based=True
    )

    # dict(
    #     type='PolyLR',
    #     power=0.9,
    #     eta_min=1e-4,
    #     begin=0,
    #     end=8,
    #     by_epoch=True
    # )
]

# optimizer

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer = dict(
#     type='SGD',  # Optimizer: Stochastic Gradient Descent
#     # lr=0.002,  # Base learning rate
#     lr=0.01,
#     # lr=0.02,
#     momentum=0.9,  # SGD with momentum
#     weight_decay=0.0005),  # Weight decay
#     # clip_grad=None
#     clip_grad=dict(max_norm=5.0, norm_type=2)
#     # clip_grad=dict(max_norm=35, norm_type=2)
#     # clip_grad=dict(max_norm=1.0, norm_type=2)

#     # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# )

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0005,
#         # weight_decay=0.05,
#         weight_decay=0.00005,
#         eps=1e-8,
#         betas=(0.9, 0.999)
#     ),
#     clip_grad=dict(max_norm=5.0, norm_type=2)
# )

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=0.0005,
        weight_decay=0.00005
    ),
    # _delete_=True,
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='Adam',
#         lr=0.0005,
#         weight_decay=0.00005
#     )
#     # clip_grad=dict(max_norm=1.0, norm_type=2)
# )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
