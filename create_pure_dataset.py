import os
from PIL import Image

# 定义路径
images_dir = "./pure_dataset_test/pre/images"
labels_dir = "./pure_dataset_test/pre/labels"
output_dir = "./pure_dataset_test/after"
class_name = 41

# 创建保存裁剪图片的目录
os.makedirs(output_dir, exist_ok=True)
i = 0
# 遍历所有图片
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)

    # 对应的标签文件
    label_name = f"{os.path.splitext(image_name)[0]}.txt"
    label_path = os.path.join(labels_dir, label_name)

    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        print(f"标签文件 {label_name} 不存在，跳过此图片")
        continue

    # 读取图片
    image = Image.open(image_path)
    image_width, image_height = image.size

    # 读取标签
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 遍历标签，处理类别为39的框
    for idx, label in enumerate(labels):
        # 解析标签内容
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])

        # 只处理类别为class_name的框
        if class_id == class_name:
            i += 1
            # 将归一化坐标转换为实际的像素坐标
            x_center = x_center * image_width
            y_center = y_center * image_height
            w = w * image_width
            h = h * image_height

            # 计算左上角和右下角的坐标
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # 确保坐标在图片范围内
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            # 裁剪图片
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image = cropped_image.convert("RGB")

            # 保存裁剪后的图片
            output_path = os.path.join(output_dir, f"{i}.jpg")
            cropped_image.save(output_path)

print(f"裁剪完成并保存所有类别为{class_name}的图片。")
