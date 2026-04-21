_base_ = './exp_template.py'

exp_id = 'ext2_cleaned_long_2k'
caption_source = 'strong'
clean_caption = True
use_short_cap = False

ext_root = '/autodl-fs/data/llmdet_extension'
ann_root = f'{ext_root}/ann_variants'
out_root = f'{ext_root}/outputs/{exp_id}'
lmm_max_token_length = 1024
num_region_caption = 8
num_sample_negative = 64

train_pipeline = _base_.train_pipeline
train_pipeline[4]['lmm_max_token_length'] = lmm_max_token_length
train_pipeline[4]['num_region_caption'] = num_region_caption
train_pipeline[4]['num_sample_negative'] = num_sample_negative

coco2017_train_dataset = dict(
    type='ODVGDataset',
    data_root='/root/autodl-tmp/grounding_data/coco/',
    ann_file=f'{ann_root}/coco/{caption_source}/instances_train2017_ext.jsonl',
    data_prefix=dict(img='train2017'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='OD',
    use_uniform_prompt=True,
    use_short_cap=use_short_cap,
    clean_caption=clean_caption,
    backend_args=None,
)

flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root='/root/autodl-tmp/grounding_data/flickr30k_entities/',
    ann_file=f'{ann_root}/flickr30k/{caption_source}/flickr_train_ext.jsonl',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_uniform_prompt=True,
    use_short_cap=use_short_cap,
    clean_caption=clean_caption,
    backend_args=None,
)

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            coco2017_train_dataset,
            flickr30k_dataset,
        ]
    )
)

model = dict(
    lmm_max_token_length=lmm_max_token_length,
    num_region_caption=num_region_caption,
)

work_dir = f'{out_root}/work_dir'

max_iter = 2000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=1000,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[1400, 1800],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=50),
)
