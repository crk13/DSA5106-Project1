# LLMDet: Learning Strong Open-Vocabulary Object Detectors

### 1. Our Experiment Environment
> **Note:** Other environments may also work, but the following configuration is tested and recommended for stability.

- **Core:** `pytorch==2.2.1+cu121`, `transformers==4.37.2`
- **Numeric:** `numpy==1.22.2` (Should be lower than 1.24; `1.23` or `1.22` is recommended)
- **Detection Ecosystem:** `mmcv==2.2.0`, `mmengine==0.10.5`
- **Dependencies:** `timm`, `deepspeed`, `pycocotools`, `lvis`, `jsonlines`, `fairscale`, `nltk`, `peft`, `wandb`

---

### 2. Model Zoo
Checkpoints and logs are available on [Hugging Face](https://huggingface.co/fushh7/LLMDet) and [ModelScope](https://modelscope.cn/models/fushh7/LLMDet).

| Model | AP<sup>mini</sup> (LVIS) | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | AP<sup>val</sup> (LVIS 1.0) | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLMDet Swin-T only p5** | 44.5 | 38.6 | 39.3 | 50.3 | 34.6 | 25.5 | 29.9 | 43.8 |
| **LLMDet Swin-T** | 44.7 | 37.3 | 39.5 | 50.7 | 34.9 | 26.0 | 30.1 | 44.3 |
| **LLMDet Swin-B** | 48.3 | 40.8 | 43.1 | 54.3 | 38.5 | 28.2 | 34.3 | 47.8 |
| **LLMDet Swin-L** | 51.1 | 45.1 | 46.1 | 56.6 | 42.0 | 31.6 | 38.8 | 50.2 |
| **LLMDet Swin-L (chunk 80)** | 52.4 | 44.3 | 48.8 | 57.1 | 43.2 | 32.8 | 40.5 | 50.8 |

---

### 3. Pretrained Models Preparation
Before running the code, ensure the following pretrained models are placed in the `huggingface` directory:

- **Foundation Models:**
  - `bert-base-uncased` (Download directly from Hugging Face)
  - `siglip-so400m-patch14-384` (Download directly from Hugging Face)
- **Multimodal Components:**
  - `my_llava-onevision-qwen2-0.5b-ov-2`: Please download our fine-tuned version from [Hugging Face](https://huggingface.co/fushh7/LLMDet).
- **Detection Backbones (MM-Grounding-DINO):**
  - [swin-t](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth)
  - [swin-b](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth)
  - [swin-l](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth)
