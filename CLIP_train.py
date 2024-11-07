import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

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

# 定义路径和创建数据加载器
train_loader = create_dataloader(r"./wds_dataset/train.tar", batch_size=32)
val_loader = create_dataloader(r"./wds_dataset/val.tar", batch_size=32)
test_loader = create_dataloader(r"./wds_dataset/test.tar", batch_size=32)

# 检查数据加载器是否正常工作
for images, texts in test_loader:
    print(images.shape, texts)  # 打印图像的张量维度和文本内容
    break


# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载 CLIP 模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')


# 定义优化器和学习率调度器
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# 定义对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07))).to(device)
        self.criterion = self.criterion.to(device)

    def forward(self, image_features, text_features):
        batch_size = len(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)  # 归一化图像特征
        text_features = F.normalize(text_features, p=2, dim=-1)  # 归一化文本特征

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        labels = torch.arange(batch_size).to(device)
        loss_image = self.criterion(logits_per_image, labels)
        loss_text = self.criterion(logits_per_text, labels)

        return (loss_image + loss_text) / 2

loss_fn = ContrastiveLoss().to(device)

# 训练过程
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=5):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0  # 手动跟踪批次数量
        for images, texts in train_loader:
            batch_count += 1
            images = images.to(device)
            texts = tokenizer(list(texts)).to(device)  # 对 texts 使用 tokenizer 进行编码，并将其移动到设备上

            optimizer.zero_grad()
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            loss = loss_fn(image_features, text_features)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / batch_count  # 使用手动跟踪的批次数量来计算平均损失
        print(f"Epoch [{epoch + 1}/{num_epochs}]: \nTraining Loss: {avg_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_batch_count = 0  # 手动跟踪验证批次数量
        with torch.no_grad():
            for images, texts in val_loader:
                val_batch_count += 1
                images = images.to(device)
                texts = tokenizer(list(texts)).to(device)  # 对 texts 使用 tokenizer 进行编码，并将其移动到设备上

                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                loss = loss_fn(image_features, text_features)

                val_loss += loss.item()

        avg_val_loss = val_loss / val_batch_count  # 使用手动跟踪的批次数量来计算平均验证损失
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        # 更新学习率
        scheduler.step()

    # 保存最佳模型权重
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), "fine_tuned_clip_model.pth")
    print("Training complete. Model saved as 'fine_tuned_clip_model.pth'")

# 开始训练
train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=30)

