# DSA5106-Project1
#### Dataset Directory Structure

```text
dataset/
├── huggingface/
│   ├── bert-base-uncased/
│   ├── siglip-so400m-patch14-384/
│   ├── my_llava-onevision-qwen2-0.5b-ov-2/
│   └── mm_grounding_dino/
│       ├── grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
│       ├── grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
│       └── grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
│
├── grounding_data/
│   ├── coco/
│   │   ├── annotations/
│   │   │   ├── instances_train2017_vg_merged6.cleaned.jsonl
│   │   │   ├── instances_val2017.json
│   │   │   ├── lvis_v1_minival_inserted_image_name.json
│   │   │   └── lvis_od_val.json
│   │   ├── train2017/
│   │   └── val2017/
│   │
│   ├── flickr30k_entities/
│   │   ├── flickr_train_vg7.cleaned.jsonl
│   │   └── flickr30k_images/
│   │
│   ├── gqa/
│   │   ├── gqa_train_vg7.cleaned.jsonl
│   │   └── images/
│   │
│   ├── llava_cap/
│   │   ├── LLaVA-ReCap-558K_tag_box_vg7.jsonl
│   │   └── images/
│   │
│   └── v3det/
│       ├── annotations/
│       │   └── v3det_available.jsonl
│       └── images/
│
└── LLMDet/
    └── configs/
        └── grounding_dino_swin_t.py
```

#### Dataset Preparation Summary

| Dataset | Images | Final Annotation | Status | Notes |
|---------|-------:|------------------|--------|-------|
| COCO train | 118287 | `instances_train2017_vg_merged6.cleaned.jsonl` | PASS | Final cleaned version used |
| COCO val | 5000 | `instances_val2017.json` | PASS | Validation split ready |
| Flickr30k | 31783 | `flickr_train_vg7.cleaned.jsonl` | PASS | Final cleaned version used |
| GQA | 148854 | `gqa_train_vg7.cleaned.jsonl` | PASS | Final cleaned version used |
| LLaVA-ReCap | 558128 | `LLaVA-ReCap-558K_tag_box_vg7.jsonl` | PASS | Original annotation retained |
| V3Det | 165465 | `v3det_available.jsonl` | PASS | Final local annotation used |

##### Image Directories

- COCO train: `grounding_data/coco/train2017`
- COCO val: `grounding_data/coco/val2017`
- Flickr30k: `grounding_data/flickr30k_entities/flickr30k_images`
- GQA: `grounding_data/gqa/images`
- LLaVA-ReCap: `grounding_data/llava_cap/images`
- V3Det: `grounding_data/v3det/images`

##### Additional Resources

- `huggingface/bert-base-uncased`
- `huggingface/siglip-so400m-patch14-384`
- `huggingface/my_llava-onevision-qwen2-0.5b-ov-2`
- `huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- `huggingface/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth`
- `huggingface/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth`
