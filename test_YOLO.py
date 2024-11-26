import torch
import open_clip
import cv2
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from ultralytics import YOLO
import numpy as np
import time

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 载入 OpenCLIP 模型和处理器
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.load_state_dict(torch.load("fine_tuned_clip_model.pth", map_location=device))
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 载入 YOLO 模型
yolo_model = YOLO('yolov10s.pt')
yolo_model.to(device)

# 定义商品类别列表
product_classes = [
    "Lemon Drink",
    "Bottle",
    "南孚电池",
    "三得利无糖乌龙茶",
    "阿萨姆奶茶",
    "六神花露水",
    "乐事黄瓜味薯片",
    "乐事原味薯片",
    "东方树叶茉莉花茶",
    "可口可乐330ml",
    "Coca Cola"
]

# 读取输入图片
image_path = 'dataset/test_CLIP_7.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

times = []
for _ in range(1):
    start_time = time.time()
    # 使用 YOLO 进行目标检测
    results = yolo_model(image)

    # 提取 YOLO 检测结果
    boxes = results[0].boxes.xyxy  # 检测框坐标
    class_ids = results[0].boxes.cls  # 类别索引
    labels = results[0].names  # 类别名称的字典

    # 将类别索引转换为实际的类别名称
    detected_labels = [labels[int(class_id)] for class_id in class_ids]

    # 对每个检测到的目标进行细类类别
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype("simsun.ttc", 15)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # 过滤框
        print((x2 - x1) * (y2 - y1))
        if (x2 - x1) * (y2 - y1) > 2000000:
            continue

        try:
            # 裁剪检测到的物体区域
            cropped_img = image_pil.crop((x1, y1, x2, y2))
            # 使用 CLIP 的预处理方法
            image_input = preprocess(cropped_img).unsqueeze(0).to(device)

            # 将商品类别列表编码为文本张量
            text_inputs = tokenizer(product_classes).to(device)

            # 编码图像和文本
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            # 计算相似度
            image_features /= image_features.norm(dim=-1, keepdim=True)     # 归一化, 使得点积等于余属相似度
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # similarity = (image_features @ text_features.T).squeeze(0).cpu().numpy()
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0).cpu().numpy()
            print(similarity)

            # 获取最佳匹配
            best_match_idx = similarity.argmax()
            best_match_score = similarity[best_match_idx]
            best_match_class = product_classes[best_match_idx]

            # 绘制检测框和标签
            label_text = f"{detected_labels[i]}->{best_match_class} ({best_match_score:.2f})"
            print(best_match_class)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            draw.text((x1 - 20, y1 - 15), label_text, font=font, fill=(36, 255, 12))

        except Exception as e:
            print(f"Error processing object {i}: {e}")
            continue
    end_time = time.time()
    times.append(end_time - start_time)

# times = times[1:]
# # 时间
# print("平均检测时间：", sum(times) / len(times))
# # 绘制消耗时间变化图
# plt.plot(times)
# plt.xlabel('Number of times')
# plt.ylabel('Time (s)')
# plt.show()

# 将 PIL 图像转换回 OpenCV 图像
image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 使用 OpenCV 显示结果
cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
cv2.imshow('Detection Results', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
