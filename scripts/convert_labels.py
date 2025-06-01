import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 设置路径
xml_dir = "../data/annotations"               # VOC XML 文件目录
label_dir = "../yolo/yolo_data/labels/"  # YOLO 标签输出目录
os.makedirs(label_dir, exist_ok=True)

xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
error_files = []

for xml_file in tqdm(xml_files, desc="Convert Labels"):
    xml_path = os.path.join(xml_dir, xml_file)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        base = os.path.splitext(filename)[0]

        # 获取图像尺寸
        size_tag = root.find("size")
        if size_tag is not None:
            w = int(size_tag.find("width").text)
            h = int(size_tag.find("height").text)
        else:
            raise ValueError(f"未找到<size>标签，文件: {xml_file}")

        label_lines = []
        for obj in root.findall("object"):
            class_id = int(obj.find("name").text)
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        label_path = os.path.join(label_dir, base + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

    except Exception as e:
        print(f" 处理失败: {xml_file}, 错误: {e}")
        error_files.append(xml_file)

# 总结输出
print(f"\n 转换完成！成功: {len(xml_files) - len(error_files)}，失败: {len(error_files)}")
if error_files:
    print("失败文件：")
    for ef in error_files:
        print(" -", ef)
