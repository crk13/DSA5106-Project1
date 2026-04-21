import os

_base_ = '../../grounding_dino_swin_t_only_p5.py'

PROJECT_ROOT = os.getenv('AUTODL_PROJECT_ROOT', '/root/project')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'tbx11k')

model = dict(
    lmm=None,
    test_cfg=dict(max_per_img=300, chunked_size=-1),
)

dataset_type = 'CocoDataset'
metainfo = dict(classes=(
    'ActiveTuberculosis',
    'ObsoletePulmonaryTuberculosis',
    'PulmonaryTuberculosis',
))

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = (
    'img_id',
    'img_path',
    'ori_shape',
    'img_shape',
    'scale_factor',
    'text',
    'custom_entities',
    'caption_prompt',
)

dataset = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=DATA_ROOT,
    ann_file='annotations/json/TBX11K_val.json',
    data_prefix=dict(img='imgs/'),
    test_mode=True,
    pipeline=base_test_pipeline,
    return_classes=True,
)

val_dataloader = dict(dataset=dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=os.path.join(DATA_ROOT, 'annotations/json/TBX11K_val.json'),
    metric='bbox',
    classwise=True,
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'],
)
test_evaluator = val_evaluator

