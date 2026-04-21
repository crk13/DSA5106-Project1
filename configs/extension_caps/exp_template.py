_base_ = '../grounding_dino_swin_t.py'

# ==============================
# Extension experiment switches
# ==============================
exp_id = 'ext_srcStrong_cleanOn_lenLong_s1'

# Source factor: use one of {strong, light, human_ref}
caption_source = 'strong'

# Cleaning factor: True = remove speculative clauses, False = raw captions
clean_caption = True

# Length/detail factor: True = short caption mode, False = long/detail mode
use_short_cap = False

# Fixed paths on storage disk (avoid system disk)
ext_root = '/autodl-fs/data/llmdet_extension'
ann_root = f'{ext_root}/ann_variants'
out_root = f'{ext_root}/outputs/{exp_id}'

# -------- Dataset annotation variants (replace with your prepared files) --------
# Keep image roots unchanged to preserve controlled settings.
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

# Keep train subset and data composition fixed unless your protocol says otherwise.
train_dataloader = dict(
    dataset=dict(
        datasets=[
            coco2017_train_dataset,
            flickr30k_dataset,
        ]
    )
)

# Put all artifacts on /autodl-fs to avoid filling '/'.
work_dir = f'{out_root}/work_dir'

# Optional: save predictions and metrics in predictable locations in test scripts.
ext_pred_path = f'{out_root}/preds/predictions.pkl'
ext_metric_path = f'{out_root}/metrics/lvis_metrics.json'
