class_weights_dict = {0: 0.04095788962337835, 1: 1.5113190636862084, 2: 0.2450592130758885, 3: 1.232093123209169, 4: 2.0934785783836416, 5: 1.7712988136431043, 6: 2.351785714285714, 7: 1.0779116113506468, 8: 0.9991646528487778, 9: 0.8343691788264515, 10: 5.764081769436998, 11: 5.42108547655068, 12: 8.365768482490273, 13: 1.092703039235617, 14: 0.9948188506385341, 15: 2.2546167156040267, 16: 1.9517088779956426, 17: 1.6320043267041142, 18: 1.1305092543905773, 19: 1.3195056462501535, 20: 1.949938781062942, 21: 8.307583075734158, 22: 2.027156798038846, 23: 2.095110602221789, 24: 1.2327995986238531, 25: 0.9404262531712011, 26: 0.8701645216124332, 27: 1.6548664562807882, 28: 1.7361131298449612, 29: 4.008207494407158, 30: 1.6175161751429432, 31: 4.00372905027933, 32: 1.6937155348983772, 33: 1.1844438629352136, 34: 3.281444597069597, 35: 2.86896517213771, 36: 1.9393852606891575, 37: 1.754817580803134, 38: 2.234000935162095, 39: 0.44162404486073453, 40: 1.358525527612789, 41: 0.5205817191283293, 42: 1.9620391494798322, 43: 1.3835279922779922, 44: 1.7437165450121654, 45: 0.7487123903050564, 46: 1.1366052548107421, 47: 1.8372949068535294, 48: 2.4582694946261148, 49: 1.6799519456165026, 50: 1.4709924055829229, 51: 1.3690795338767192, 52: 3.684034441398218, 53: 1.8467638721869095, 54: 1.4974247806101129, 55: 1.6921159294821344, 56: 0.27928639162401603, 57: 1.8601855857414777, 58: 1.242488730929265, 59: 2.5644113788167937, 60: 0.6841041428025965, 61: 2.5860025258599952, 62: 1.8518540051679586, 63: 2.1629803822937625, 64: 4.75243700265252, 65: 1.8849750131509733, 66: 3.765328371278459, 67: 1.6708132576935033, 68: 6.425590257023312, 69: 3.224358878224355, 70: 47.777833333333334, 71: 1.9162232620320856, 72: 4.0766069397042095, 73: 0.43495903297592553, 74: 1.6971917429744237, 75: 1.6255878572508695, 76: 7.25861748818366, 77: 2.242856770290006, 78: 54.29299242424243, 79: 5.501541709314227, 80: 0.01}

# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=2,
        frozen_stages=1,
        # frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),


    # roi_head=dict(
    #     type='StandardRoIHead',
    #     bbox_roi_extractor=dict(
    #         type='SingleRoIExtractor',
    #         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
    #         out_channels=256,
    #         featmap_strides=[4, 8, 16, 32]
    #     ),
    #     bbox_head=dict(
    #         type='Shared2FCBBoxHead',
    #         in_channels=256,
    #         fc_out_channels=1024,
    #         roi_feat_size=7,
    #         num_classes=80,
    #         bbox_coder=dict(
    #             type='DeltaXYWHBBoxCoder',
    #             target_means=[0., 0., 0., 0.],
    #             target_stds=[0.1, 0.1, 0.2, 0.2]
    #         ),
    #         reg_class_agnostic=False,
    #         # loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_cls=dict(type='CrossEntropyRoILoss', use_sigmoid=False, loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    #     )
    # ),

    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCHyperbolicBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            # fc_out_channels=512,
            # fc_out_channels=100,
            # fc_out_channels=2,
            roi_feat_size=7,
            num_classes=80,
            # num_classes=91,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='PeBusePenaltyLoss',
                dimension=2,
                num_classes=80,
                prototype_path = "/home/amakacs1/mmdetection/hyperbolic_assets/prototypes/",
                penalty_option='dim',
                # mult=1.0,
                mult=0.75,
                # mult=0.1,
                loss_weight=1.0
            ),

            # loss_cls=dict(
            #     type='WeightedBusePenaltyLoss',
            #     dimension=100,
            #     num_classes=80,
            #     prototype_path = "/home/amakacs1/mmdetection/hyperbolic_assets/prototypes/",
            #     penalty_option='dim',
            #     # mult=1.0,
            #     mult=0.75,
            #     # mult=0.1,
            #     loss_weight=1.0,
            #     class_weights=class_weights_dict
            # ),

            # loss_cls=dict(type='CosineLoss', dimension=50),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),

    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                # num=256,
                # num=128,
                # num=64,
                # num=32,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            # nms_pre=500,
            # max_per_img=500,
            # nms_pre=50,
            # max_per_img=50,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        # rcnn=None
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)
