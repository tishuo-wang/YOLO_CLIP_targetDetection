import os
import tarfile
import shutil
from tqdm import tqdm

# 数据集路径
root_dir = "./pure_dataset_test/split_dataset"
output_dir = "./wds_dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取数据集并将其打包为 WebDataset 格式的 tar 文件
def create_webdataset(root_dir, split_name):
    split_dir = os.path.join(root_dir, split_name)
    output_tar = os.path.join(output_dir, f"{split_name}.tar")

    with tarfile.open(output_tar, 'w') as tar:
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
                    image_path = os.path.join(class_dir, image_name)
                    if image_path.endswith(('.png', '.jpg', '.jpeg')):
                        # 添加图像文件
                        arcname_image = f"{class_name}_{image_name}"
                        tar.add(image_path, arcname=arcname_image)

                        # 创建文本描述文件
                        text_description = f"{class_name}"
                        txt_path = os.path.join(output_dir, f"{arcname_image}.txt")
                        # 使用 UTF-8 编码写入文本文件
                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(text_description)

                        # 将文本文件添加到 tar 中
                        tar.add(txt_path, arcname=f"{arcname_image}.txt")

                        # 删除临时文本文件
                        os.remove(txt_path)

# 处理 train、val 和 test 数据集
splits = ['train', 'val', 'test']
for split in splits:
    create_webdataset(root_dir, split)

# 删除临时文件夹
# shutil.rmtree(output_dir)




# from webdataset.gopen import gopen
#
# url = "file://D:/CODE/PyTorch/PyTorch_project1/chuanxinshiijan_test/CLIP_train_test/wds_dataset/train.tar"
#
# try:
#     with gopen(url, "rb") as stream:
#         print("File opened successfully")
# except ValueError as e:
#     print(f"ValueError: {e}")
# except OSError as e:
#     print(f"OSError: {e}")














