import os
import json
from PIL import Image

# 定义输入和输出目录
input_root_dir = "pure_dataset_test/pre"
output_root_dir = "pure_dataset_test/after"

# 确保输出根目录存在
os.makedirs(output_root_dir, exist_ok=True)

# 遍历类别目录
for category_name in os.listdir(input_root_dir):
    category_dir = os.path.join(input_root_dir, category_name)

    # 确保目录下有 images 和 labels 子文件夹
    images_dir = os.path.join(category_dir, "images")
    labels_dir = os.path.join(category_dir, "Annotations")
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"目录 {category_name} 缺少 images 或 labels 子文件夹，跳过此类别")
        continue

    # 定义输出目录
    output_category_dir = os.path.join(output_root_dir, category_name)
    os.makedirs(output_category_dir, exist_ok=True)

    # 加载 JSON 文件
    json_path = os.path.join(labels_dir, "coco_info.json")
    if not os.path.exists(json_path):
        print(f"类别 {category_name} 缺少标签文件 coco_info.json，跳过此类别")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建一个字典，将图片的 id 和文件名对应起来
    image_dict = {image['id']: image for image in data['images']}
    i = 0

    # 遍历 annotations 中的每一个 bbox 信息
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        if image_id not in image_dict:
            print(f"未找到 id 为 {image_id} 的图像，跳过此标注")
            continue

        # 获取图片文件名和路径
        image_info = image_dict[image_id]
        image_name = image_info['file_name']
        image_path = os.path.join(images_dir, image_name)

        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件 {image_name} 不存在，跳过此标注")
            continue

        # 打开图片
        image = Image.open(image_path)

        # 提取边界框的坐标并确保在图像边界范围内
        x, y, w, h = bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image.width, int(x + w))
        y2 = min(image.height, int(y + h))

        # 确保裁剪区域的有效性（宽高都大于零）
        if x2 <= x1 or y2 <= y1:
            print(f"无效的裁剪区域 {bbox}，跳过此标注")
            continue

        # 裁剪图片
        cropped_image = image.crop((x1, y1, x2, y2))

        # 将裁剪后的图片转换为 RGB 格式，以防止透明度问题
        cropped_image = cropped_image.convert("RGB")

        # 保存裁剪后的图片
        i += 1
        output_path = os.path.join(output_category_dir, f"{i}.jpg")
        cropped_image.save(output_path)

print("处理完成。")
