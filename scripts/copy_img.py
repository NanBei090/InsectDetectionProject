import os
import shutil
from tqdm import tqdm

# 路径设置
label_dir = "../yolo/yolo_data/labels/"  # YOLO标签路径（.txt 文件）
image_src_dir = "../data/JPEGImages/"       # 所有原始图片所在目录（需修改为你实际目录）
image_dst_dir = "../yolo/yolo_data/images/"  # 目标路径（复制到这里）
os.makedirs(image_dst_dir, exist_ok=True)

# 允许的图片格式（可扩展）
img_exts = [".jpg", ".jpeg", ".png", ".bmp"]

# 读取标签文件名（不含扩展名）
label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
basenames = [os.path.splitext(f)[0] for f in label_files]

missing = []

for base in tqdm(basenames, desc="Copying images"):
    found = False
    for ext in img_exts:
        img_file = base + ext
        src_path = os.path.join(image_src_dir, img_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(image_dst_dir, img_file)
            shutil.copy2(src_path, dst_path)
            found = True
            break
    if not found:
        missing.append(base)

# 总结
print(f"\n图片复制完成！共复制: {len(basenames) - len(missing)} 张图像。")
if missing:
    print(f" 找不到对应图像: {len(missing)} 个")
    for m in missing:
        print(" -", m)
