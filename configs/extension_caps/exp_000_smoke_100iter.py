_base_ = './exp_001_strong_cleanOn_lenLong.py'

# Quick sanity-check run before full training.
exp_id = 'exp_000_smoke_100iter'
out_root = f'/autodl-fs/data/llmdet_extension/outputs/{exp_id}'
work_dir = f'{out_root}/work_dir'

max_iter = 100
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=100,
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=20),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[70, 90],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=100, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=10),
)
