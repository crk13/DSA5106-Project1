# LLMDet Extension Workflow (Caption Causal Study)

This folder is for your extension study implementation.

## 1) Where to put files

- Large data and experiment outputs:
  - /autodl-fs/data/llmdet_extension
- Lightweight code/config in repo:
  - configs/extension_caps
  - scripts/extension
  - extension

## 2) Factors mapped to existing code switches

- Caption source (strong/light/human_ref):
  - Implement by preparing separate JSONL annotation variants.
  - Switch by changing ann_file paths in config.
- Caption cleaning (hallucination control):
  - clean_caption = True/False
- Caption length/detail:
  - use_short_cap = True/False

These switches are already supported by ODVGDataset in mmdet/datasets/odvg.py.

## 3) Recommended storage layout

- /autodl-fs/data/llmdet_extension/captions/source_strong
- /autodl-fs/data/llmdet_extension/captions/source_light
- /autodl-fs/data/llmdet_extension/captions/source_human_ref
- /autodl-fs/data/llmdet_extension/ann_variants
- /autodl-fs/data/llmdet_extension/outputs/<exp_id>/{work_dir,preds,metrics,analysis}

## 4) Minimal run steps

1. Copy and edit configs/extension_caps/exp_template.py:
   - set exp_id
   - set caption_source
   - set clean_caption
   - set use_short_cap
   - set ann_file paths to your prepared JSONL variants
2. Run:
   - bash scripts/extension/run_extension_exp.sh configs/extension_caps/exp_template.py 1 none
3. Evaluate with existing test command and save logs under
   /autodl-fs/data/llmdet_extension/outputs/<exp_id>/metrics

## 5) Do you need a notebook?

Notebook is optional.

- Not required for training/evaluation.
- Recommended only for analysis/plotting (AP/APr/APc/APf comparison).

A simple Python script is usually enough for batch experiments.
