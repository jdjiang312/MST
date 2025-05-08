weight = None
resume = True
evaluate = True
test_only = True
num_worker = 40
batch_size = 8
batch_size_val = None
batch_size_test = None
seed= None

epoch = 800
eval_epoch = 800
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = True
mix_prob = 0.8

hooks = [
    dict(type='CheckpointLoader', keywords='module.', replacement='module.'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(
        type='InsSegEvaluator',
        segment_ignore_index=(-1, ),
        instance_ignore_index=-1),
    dict(type='CheckpointSaver', save_freq=None)
]

train = dict(type='MultiDatasetTrainer')
test = dict(type='InsSegTester', verbose=True)

class_names = ['tree']
num_classes = 1
segment_ignore_index = (-1, )

model = dict(
    type='PG-v1m1',
    backbone=dict(
        type='MST-v1m1',
        backbone=dict(
            type='PT-v3m1',
            in_channels=3,
            order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
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
            dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1)
        ],
        backbone_out_channels=64,
        context_channels=256,
        conditions=('S3DIS', 'S3DIS', 'S3DIS'),
        template='[x]',
        clip_model='ViT-B/16',
        class_name=('tree', ),
        valid_index=((0, ), (0, ), (0, )),
        backbone_mode=True),
    backbone_out_channels=64,
    semantic_num_classes=1,
    semantic_ignore_index=-1,
    segment_ignore_index=(-1, ),
    instance_ignore_index=-1,
    cluster_thresh=2.5,
    cluster_closed_points=300,
    cluster_propose_points=500,
    cluster_min_points=100,
    voxel_size=0.04)

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
    num_classes=1,
    ignore_index=-1,
    names=['tree'],
    train=dict(
        type='ConcatDataset',
        datasets=[
            ## ALS dataset
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
                        return_grid_coord=True,
                        keys=('coord', 'segment', 'instance')),
                    dict(type='SphereCrop', sample_rate=0.8, mode='random'),
                    dict(
                        type='InstanceParser',
                        segment_ignore_index=(-1, ),
                        instance_ignore_index=-1),
                    dict(type='Add', keys_dict=dict(condition='S3DIS')),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 
                              'grid_coord', 
                              'segment', 
                              'instance',
                              'instance_centroid', 
                              'bbox', 
                              'condition'),
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
                        return_grid_coord=True,
                        keys=('coord', 'segment', 'instance')),
                    dict(type='SphereCrop', sample_rate=0.8, mode='random'),
                    dict(
                        type='InstanceParser',
                        segment_ignore_index=(-1, ),
                        instance_ignore_index=-1),
                    dict(type='Add', keys_dict=dict(condition='S3DIS')),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 
                              'grid_coord', 
                              'segment', 
                              'instance',
                              'instance_centroid', 
                              'bbox', 
                              'condition'),
                        feat_keys=('coord', ))
                ],
                test_mode=False,
                loop=1),
            
            ## MLS dataset
            dict(
                type='S3DISDataset',
                split=('train'),
                data_root='/MLS/datasets/path',
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
                        return_grid_coord=True,
                        keys=('coord', 'segment', 'instance')),
                    dict(type='SphereCrop', sample_rate=0.8, mode='random'),
                    dict(
                        type='InstanceParser',
                        segment_ignore_index=(-1, ),
                        instance_ignore_index=-1),
                    dict(type='Add', keys_dict=dict(condition='S3DIS')),
                    dict(type='ToTensor'),
                    dict(
                        type='Collect',
                        keys=('coord', 
                              'grid_coord', 
                              'segment', 
                              'instance',
                              'instance_centroid', 
                              'bbox', 
                              'condition'),
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
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                keys=('coord', 'segment', 'instance')),
            dict(type='CenterShift', apply_z=False),
            dict(
                type='InstanceParser',
                segment_ignore_index=(-1, ),
                instance_ignore_index=-1),
            dict(type='Add', keys_dict=dict(condition='S3DIS')),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 
                      'grid_coord', 
                      'segment', 
                      'instance',
                      'origin_coord', 
                      'origin_segment', 
                      'origin_instance',
                      'instance_centroid', 
                      'bbox', 
                      'condition'),
                feat_keys=('coord', ),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False),

    
    test=dict(
        type='S3DISDataset',
        split='test',
        data_root='/your/test/path',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy',
                 keys_dict=dict(coord='origin_coord', segment='origin_segment', instance='origin_instance')),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'segment', 'instance'),
                return_grid_coord=True,
                return_inverse=True
            ),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(
                    type='InstanceParser',
                    segment_ignore_index=(-1,),
                    instance_ignore_index=-1),
                dict(type='Add', keys_dict=dict(condition=['S3DIS'])),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 
                          'grid_coord', 
                          'segment', 
                          'instance', 
                          'inverse',
                          'origin_coord', 
                          'origin_segment', 
                          'origin_instance',
                          'instance_centroid', 
                          'bbox', 
                          'condition'),
                    feat_keys=('coord',),
                    offset_keys_dict=dict(
                        offset='coord',
                        origin_offset='origin_coord'
                    )
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1.0, 1.0])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1.0, 1.0]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        )
    )
)

## train
# python tools/train.py --config-file configs/pre_train/insseg_pointgroup_mst.py --options save_path=exp/MST_ins 

# test on best model
# python tools/test.py --config-file configs/pre_train/insseg_pointgroup_mst_test.py --options save_path=exp/MST_ins weight=exp/MST_ins/model/model_best.pth
