import os
import json

# 模拟配置文件中的路径拼接
data_root = '../grounding_data/coco/'
ann_file = 'annotations/lvis_v1_minival_inserted_image_name.json'
full_path = os.path.join(data_root, ann_file)

if os.path.exists(full_path):
    print(f"✅ 成功！验证集标注文件已就绪: {full_path}")
    # 尝试读取一下确认文件没损坏
    with open(full_path, 'r') as f:
        data = json.load(f)
        print(f"📊 验证集包含类别数: {len(data['categories'])}")
else:
    print(f"❌ 错误：找不到文件。请确认你是否在 LLMDet 代码文件夹内运行此脚本，且文件存放在 {full_path}")