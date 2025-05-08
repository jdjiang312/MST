weight = None
resume = False
evaluate = True
test_only = False
seed = None

num_worker = 40
batch_size = 8
batch_size_val = None
batch_size_test = None
epoch = 800
eval_epoch = 800

sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = True
mix_prob = 0.8

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]

train = dict(type='MultiDatasetTrainer')
test = dict(type='SemSegTester', verbose=True)
clip_grad = 3.0

model = dict(
    type='MST-v1m1',
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 6, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        CIM_bn=True,
        CIM_ln=True,
        CIM_decouple=True,
        CIM_adaptive=False,
        CIM_affine=True,
        CIM_conditions=('S3DIS', 'S3DIS', 'S3DIS')),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ],
    backbone_out_channels=64,
    context_channels=256,
    conditions=('S3DIS', 'S3DIS', 'S3DIS'),
    template='[x]',
    clip_model='ViT-B/16',
    class_name=('understory', 'terrain', 'leaf', 'wood'),
    valid_index=((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)),
    backbone_mode=False)

param_dicts = [dict(keyword='block', lr=0.0005)]
optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.005, 0.0005],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)

data = dict(
    num_classes=4,
    ignore_index=-1,
    names=['understory', 'terrain', 'leaf', 'wood'],
    train=dict(
        type='ConcatDataset',
        ## ALS dataset
        datasets=[
            dict(
                type='S3DISDataset',
                split=('train'),
                data_root='/ALS/dataset/path',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RandomDropout',
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2),
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='x',
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='y',
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='ElasticDistortion',
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type='ChromaticAutoContrast', p=0.2,
                        blend_factor=None),
                    dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
                    dict(type='ChromaticJitter', p=0.95, std=0.05),
                    dict(
                        type='GridSample',
                        grid_size=0.02,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', sample_rate=0.6, mode='random'),
                    dict(type='SphereCrop', point_max=204800, mode='random'),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='ShufflePoint'),
                    dict(type='Add', keys_dict=dict(condition='S3DIS')),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment', 'condition'),
                        feat_keys=('coord', ))
                ],
                test_mode=False,
                loop=1),

            ## ULS dataset
            dict(
                type='S3DISDataset',
                split=('train'),
                data_root='/ULS/dataset/path',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RandomDropout',
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2),
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='x',
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='y',
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='ElasticDistortion',
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type='ChromaticAutoContrast', p=0.2,
                        blend_factor=None),
                    dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
                    dict(type='ChromaticJitter', p=0.95, std=0.05),
                    dict(
                        type='GridSample',
                        grid_size=0.02,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', sample_rate=0.6, mode='random'),
                    dict(type='SphereCrop', point_max=204800, mode='random'),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='ShufflePoint'),
                    dict(type='Add', keys_dict=dict(condition='S3DIS')),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment', 'condition'),
                        feat_keys=('coord', ))
                ],
                test_mode=False,
                loop=1),

            ## MLS dataset
            dict(
                type='S3DISDataset',
                split=('train'),
                data_root='/MLS/dataset/path',
                transform=[
                    dict(type='CenterShift', apply_z=True),
                    dict(
                        type='RandomDropout',
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2),
                    dict(
                        type='RandomRotate',
                        angle=[-1, 1],
                        axis='z',
                        center=[0, 0, 0],
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='x',
                        p=0.5),
                    dict(
                        type='RandomRotate',
                        angle=[-0.015625, 0.015625],
                        axis='y',
                        p=0.5),
                    dict(type='RandomScale', scale=[0.9, 1.1]),
                    dict(type='RandomFlip', p=0.5),
                    dict(type='RandomJitter', sigma=0.005, clip=0.02),
                    dict(
                        type='ElasticDistortion',
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type='ChromaticAutoContrast', p=0.2,
                        blend_factor=None),
                    dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
                    dict(type='ChromaticJitter', p=0.95, std=0.05),
                    dict(
                        type='GridSample',
                        grid_size=0.02,
                        hash_type='fnv',
                        mode='train',
                        return_grid_coord=True),
                    dict(type='SphereCrop', sample_rate=0.6, mode='random'),
                    dict(type='SphereCrop', point_max=204800, mode='random'),
                    dict(type='CenterShift', apply_z=False),
                    dict(type='ShufflePoint'),
                    dict(type='Add', keys_dict=dict(condition='S3DIS')),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 'grid_coord', 'segment', 'condition'),
                        feat_keys=('coord', ))
                ],
                test_mode=False,
                loop=1)
        ],
        loop=1),
    
    val=dict(
        type='S3DISDataset',
        split='val',
        data_root='/your/validation/path',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='ToTensor'),
            dict(type='Add', keys_dict=dict(condition='S3DIS')),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'condition'),
                feat_keys=('coord', ))
        ],
        test_mode=False),
    
    test=dict(
        type='S3DISDataset',
        split='test',
        data_root='/your/test/path',
        transform=[dict(type='CenterShift', apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                keys=('coord', ),
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='Add', keys_dict=dict(condition='S3DIS')),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index', 'condition'),
                    feat_keys=('coord', ))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }], [{
                               'type': 'RandomFlip',
                               'p': 1
                           }]])))

## train
# python tools/train.py --config-file configs/pre_train/semseg_ptv3_mst.py --options save_path=exp/MST_sem

# test on best model
# python tools/test.py --config-file configs/pre_train/insseg_pointgroup_mst_test.py --options save_path=exp/MST_ins weight=exp/MST_ins/model/model_best.pth

