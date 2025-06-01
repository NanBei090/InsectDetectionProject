import os
import shutil
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

img_dir = "../yolo_data/images/"
lbl_dir = "../yolo_data/labels/"

# 获取所有图片路径
all_images = sorted(glob(os.path.join(img_dir, "*.jpg")))

# 8:2划分训练集和验证集
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

# 定义目标目录
final_img_train = os.path.join(img_dir, "train")
final_img_val = os.path.join(img_dir, "val")
final_lbl_train = os.path.join(lbl_dir, "train")
final_lbl_val = os.path.join(lbl_dir, "val")

for subset, image_list, img_dst, lbl_dst in [
    ("train", train_imgs, final_img_train, final_lbl_train),
    ("val", val_imgs, final_img_val, final_lbl_val)
]:
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    for img_path in tqdm(image_list, desc=f"Split {subset}"):
        base = os.path.basename(img_path)
        label_path = os.path.join(lbl_dir, base.replace(".jpg", ".txt"))
        dst_img = os.path.join(img_dst, base)
        dst_lbl = os.path.join(lbl_dst, base.replace(".jpg", ".txt"))
        if os.path.exists(img_path):
            shutil.move(img_path, dst_img)
        if os.path.exists(label_path):
            shutil.move(label_path, dst_lbl)
print(f"训练集图片数量: {len(train_imgs)}，标签数量: {len(final_lbl_train)}")
print(f"验证集图片数量: {len(val_imgs)}，标签数量: {len(final_lbl_val)}")
print("数据集8:2划分并移动到最终目录完成。")