import os
import random
import shutil
from tqdm import tqdm

# 数据输入和输出目录
input_dir = "pure_dataset_test/after"
output_dir = "pure_dataset_test/split_dataset"

# 水平创建输出目录
for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

# 设置每个分类的分割比例
train_ratio = 0.8
val_ratio = 0.15

# 遍历每个类别文件夹
for category in tqdm(os.listdir(input_dir), desc="Processing categories"):
    category_path = os.path.join(input_dir, category)
    if not os.path.isdir(category_path):
        continue

    # 获取所有文件
    all_files = os.listdir(category_path)
    random.shuffle(all_files)

    # 分割数据集
    train_split = int(len(all_files) * train_ratio)
    val_split = int(len(all_files) * (train_ratio + val_ratio))

    train_files = all_files[:train_split]
    val_files = all_files[train_split:val_split]
    test_files = all_files[val_split:]

    # 分别移动文件到各自分类目录
    for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        split_category_path = os.path.join(output_dir, split, category)
        os.makedirs(split_category_path, exist_ok=True)

        for file_name in tqdm(files, desc=f"Processing {category} - {split}", leave=False):
            src_file = os.path.join(category_path, file_name)
            dst_file = os.path.join(split_category_path, file_name)
            shutil.copy(src_file, dst_file)

print("数据集分割完成。")
