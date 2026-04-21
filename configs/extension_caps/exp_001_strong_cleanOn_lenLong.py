_base_ = '../grounding_dino_swin_t.py'

# Fixed first extension run for pipeline sanity check.
exp_id = 'exp_001_strong_cleanOn_lenLong'
caption_source = 'strong'
clean_caption = True
use_short_cap = False

ext_root = '/autodl-fs/data/llmdet_extension'
ann_root = f'{ext_root}/ann_variants'
out_root = f'{ext_root}/outputs/{exp_id}'

coco2017_train_dataset = dict(
    ann_file=f'{ann_root}/coco/{caption_source}/instances_train2017_ext.jsonl',
    use_short_cap=use_short_cap,
    clean_caption=clean_caption,
)

flickr30k_dataset = dict(
    ann_file=f'{ann_root}/flickr30k/{caption_source}/flickr_train_ext.jsonl',
    use_short_cap=use_short_cap,
    clean_caption=clean_caption,
)

train_dataloader = dict(
    dataset=dict(
        datasets=[
            coco2017_train_dataset,
            flickr30k_dataset,
        ]
    )
)

work_dir = f'{out_root}/work_dir'
ext_pred_path = f'{out_root}/preds/predictions.pkl'
ext_metric_path = f'{out_root}/metrics/lvis_metrics.json'
