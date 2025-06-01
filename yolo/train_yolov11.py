from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")

    model.train(
        data="./ip102.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name="yolo11s-vehicle",
        project="runs/train",
        workers=4,
        save_period=-1,  # 只保存最佳权重
        val=True,
        save=True,
        exist_ok=True,   # 允许覆盖已有目录
    )

    # 通过model.best获取最佳权重路径，避免手动拼接路径错误
    best_model_path = model.best
    print(f"最佳模型路径：{best_model_path}")

    # 用最佳权重重新加载模型
    model = YOLO(best_model_path)

    # 评估验证集
    metrics = model.val()
    print("验证指标：", metrics)

    # 导出模型（ONNX）
    export_path = model.export(format="onnx")
    print(f"模型已导出至: {export_path}")

if __name__ == "__main__":
    main()
