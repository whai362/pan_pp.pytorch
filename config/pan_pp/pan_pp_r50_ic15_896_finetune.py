model = dict(
    type='PAN_PP',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(256, 512, 1024, 2048),
        out_channels=128
    ),
    detection_head=dict(
        type='PAN_PP_DetHead',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False,
    ),
    recognition_head=dict(
        type='PAN_PP_RecHead',
        input_dim=512,
        hidden_dim=128,
        feature_size=(8, 32)
    )
)
data = dict(
    batch_size=16,
    # train=dict(
    #     type='PAN_PP_Joint_Train',
    #     split='train',
    #     is_transform=True,
    #     img_size=896,
    #     short_size=896,
    #     kernel_scale=0.5,
    #     read_type='pil',
    #     with_rec=True
    # ),
    test=dict(
        type='PAN_PP_IC15',
        split='test',
        short_size=896,
        read_type='pil',
        with_rec=True
    )
)
# train_cfg = dict(
#     lr=1e-3,
#     schedule='polylr',
#     epoch=3,
#     optimizer='Adam'
# )
test_cfg = dict(
    min_score=0.8,
    min_area=260,
    min_kernel_area=2.6,
    scale=4,
    bbox_type='rect',
    result_path='outputs/submit_ic15_rec.zip',
    rec_post_process=dict(
        len_thres=3,
        score_thres=0.95,
        unalpha_score_thres=0.9,
        ignore_score_thres=0.90,
        edit_dist_thres=2,
        voc_type=None,
        voc_path=None
    )
)
