_base_ = '../grounding_dino_swin_t.py'

model = dict(
    lmm=None,
)

dataset_type = 'LVISV1Dataset'
data_root = '/root/autodl-tmp/grounding_data/coco/'

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root + 'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator
