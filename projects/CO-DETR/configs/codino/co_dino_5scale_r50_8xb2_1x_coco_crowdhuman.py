_base_ = './co_dino_5scale_r50_lsj_8xb2_1x_coco.py'

num_classes = 1

model = dict(
    use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

# dataset settings
data_root = '/home/tao/dataset/crowdhuman/'
dataset_type = 'CrowdHumanDataset'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation_train.odgt',
        data_prefix=dict(img='CrowdHuman_train/Images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline,
                                    type=dataset_type,
                                    data_root=data_root,
                                    ann_file='annotation_val.odgt',
                                    data_prefix=dict(img='CrowdHuman_val/Images/')))

val_evaluator = dict(
    _delete_=True,
    type='mmdet.CrowdHumanMetric',
    ann_file=data_root + 'annotation_val.odgt',
    metric=['AP', 'MR', 'JI'])
test_evaluator = val_evaluator 