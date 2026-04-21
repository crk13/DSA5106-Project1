# Extension1 Evaluation Code README

## 1. Scope
This package contains only the code used for **Extension1 evaluation** (trained checkpoint vs. base/original checkpoint) on:
- ODINW-as-COCO
- Brain Tumor (cleaned annotation)
- TBX11K (val split)

It does **not** include datasets, checkpoints, or output logs.

---

## 2. Included Files

```text
code/
  mmdet_test.py
  autodl/
    run_full_eval.sh
    collect_eval_results.py
  configs/val/custom_autodl/
    grounding_dino_swin-t_only_p5_eval_odinw_custom.py
    grounding_dino_swin-t_only_p5_eval_brain_tumor_custom.py
    grounding_dino_swin-t_only_p5_eval_tbx11k_custom.py
```

---

## 3. Expected Project Layout on Server

```text
/root/project/
  code/                    # this submitted code folder
  checkpoints/
    xxx.pth                # model checkpoint to evaluate
  data/
    odinw/
    brain_tumor/
    tbx11k/
  work_dirs/               # auto-created
```

If your project root is not `/root/project`, set environment variable:

```bash
export AUTODL_PROJECT_ROOT=/your/project/root
```

---

## 4. Environment Requirements

Please ensure the runtime can import:
- torch
- mmengine
- mmcv
- mmdet
- transformers
- pycocotools

and `torch.cuda.is_available()` is `True`.

---

## 5. Run Full Evaluation (3 datasets)

From project root:

```bash
cd /root/project/code
bash autodl/run_full_eval.sh /root/project/checkpoints/your_model.pth
```

This script runs:
1. ODINW
2. Brain Tumor
3. TBX11K

and then collects metrics into summary files.

---

## 6. Output Files

A new folder is generated at:

```text
/root/project/work_dirs/eval_full_YYYYMMDD_HHMMSS/
```

Main outputs:
- `odinw/`, `brain_tumor/`, `tbx11k/` subfolders
- `summary_full.json`
- `summary_full.csv`

---

## 7. Notes

1. Brain Tumor config uses cleaned annotation:
   - `data/brain_tumor/test/_annotations.cleaned.coco.json`
2. TBX11K config uses:
   - `data/tbx11k/annotations/json/TBX11K_val.json`
3. By default the script dumps `preds.pkl` for each dataset (may be large).

If needed, you can remove old prediction files after metric collection to save disk space.
