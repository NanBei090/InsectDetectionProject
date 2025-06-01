from ultralytics import YOLO

def main():
    # 加载预训练模型权重，这里用的是yolo11s的小模型
    model = YOLO("yolo11s.pt")

    # 开始训练
    model.train(
        data="./ip102.yaml",       # 数据集配置文件路径，定义了训练/验证集等信息
        epochs=200,                # 最大训练周期数
        imgsz=640,                 # 输入图片尺寸（宽高），保持方形
        batch=16,                  # 每个批次的样本数量
        device=0,                  # 使用的设备编号，0表示第一块GPU，若无GPU可用"cpu"
        name="yolo11s-vehicle",    # 训练输出子目录名称，保存在 project 参数目录下
        project="runs",            # 保存训练结果的根目录
        workers=4,                 # 加载数据时的线程数，提高数据读取速度
        save_period=-1,            # 只保存最佳权重，不保存每N轮权重
        val=True,                  # 每个训练周期结束后进行验证
        save=True,                 # 保存模型权重文件
        exist_ok=True,             # 如果目录已存在则允许覆盖
    )

    # 获取训练过程中保存的最佳模型权重路径，避免手动拼接错误
    best_model_path = model.best
    print(f"最佳模型路径：{best_model_path}")

    # 使用最佳权重重新加载模型
    model = YOLO(best_model_path)

    # 在验证集上评估模型性能，返回指标
    metrics = model.val()
    print("验证指标：", metrics)

    # 导出模型为ONNX格式，方便部署到其他平台
    export_path = model.export(format="onnx")
    print(f"模型已导出至: {export_path}")

if __name__ == "__main__":
    main()
