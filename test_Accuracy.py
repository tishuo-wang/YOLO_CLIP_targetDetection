import webdataset as wds
from torch.utils.data import DataLoader
import open_clip
import torch
import torch.nn.functional as F
from torchvision import transforms

# 定义基本的图像预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 定义文本处理函数（保持原样）
def identity(x):
    return x

# 加载 WebDataset 数据
def create_dataloader(tar_path, batch_size):
    dataset = (
        wds.WebDataset(tar_path)
        .shuffle(1000)  # 在这里打乱数据，数字表示缓冲区大小
        .decode("pil")  # 解码为 PIL 格式
        .to_tuple("jpg;jpeg;png", "jpg.txt;jpeg.txt;png.txt")  # 从 .tar 文件中提取图像和对应文本
        .map_tuple(transform, identity)  # 对图像进行预处理，对文本保持原样
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)  # 注意：设置 num_workers=0 以避免多进程问题
    return dataloader

# 创建测试集的数据加载器
test_loader = create_dataloader(r"./wds_dataset/test.tar", batch_size=32)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载已训练的 CLIP 模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.load_state_dict(torch.load("fine_tuned_clip_model.pth", map_location=device))
model = model.to(device)
model.eval()

# 加载 tokenizer
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 定义商品类别列表（与训练期间相同）
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

# 定义测试函数
def evaluate_model(model, dataloader, tokenizer, product_classes):
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, texts in dataloader:
            # 将图像和文本移动到设备上
            images = images.to(device)
            texts = [text.strip() for text in texts]

            # 预处理文本并转换为文本张量
            text_inputs = tokenizer(product_classes).to(device)

            # 编码图像和文本
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_inputs)

            # 计算相似度
            image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # 获取预测结果
            predicted_indices = similarity.argmax(dim=-1).cpu().numpy()

            # 计算准确率
            for idx, predicted_idx in enumerate(predicted_indices):
                predicted_class = product_classes[predicted_idx]
                true_class = texts[idx]
                if predicted_class == true_class:
                    print(true_class)
                    correct_predictions += 1
                total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")

# 运行测试
evaluate_model(model, test_loader, tokenizer, product_classes)
