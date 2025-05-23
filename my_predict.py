# my_first_prediction_script.py

from ultralytics import YOLO

# 1. 加载模型
# Ultralytics 会自动下载模型到 ~/.cache/ultralytics/models 或其他默认位置
# 如果你确定要使用YOLOv11，可以尝试 'yolov11n.pt'，但通常YOLOv8模型是更通用的起点。
# 我们先用YOLOv8n.pt，因为它稳定且项目兼容。
model = YOLO('yolov8n.pt')

# 2. 进行预测
# source: 可以是图片路径、视频路径、URL、摄像头索引 (0)
# save: 是否保存结果图片/视频到 runs/detect/predict 目录 (会在项目根目录自动创建 runs/detect/)
# show: 是否实时显示预测窗口 (对于视频和摄像头)
# conf: 置信度阈值，低于此阈值的检测框不显示
print("Starting prediction...")
results = model.predict(
    source='https://ultralytics.com/images/bus.jpg',  # 使用一个网络图片进行测试
    # source='path/to/your/local_image.jpg',  # 如果使用本地图片，请替换为实际路径
    save=True,
    show=False,  # 在服务器或无图形界面下通常设为 False
    conf=0.25  # 较低的置信度阈值，更容易看到检测结果
)
print("Prediction complete.")

# 3. (可选) 处理并打印预测结果的详细信息
print("\n--- Detailed Prediction Results ---")
for i, result in enumerate(results):
    print(f"Result for item {i + 1}:")
    if result.boxes:
        print(f"  Detected {len(result.boxes)} bounding boxes.")
        for j, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox_coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2] format
            class_name = model.names[class_id]  # 获取类别名称
            print(f"    Box {j + 1}: Class='{class_name}', Conf={confidence:.2f}, Coords={bbox_coords}")

    if result.masks:
        print(f"  Detected {len(result.masks)} segmentation masks.")

    if result.keypoints:
        print(f"  Detected {len(result.keypoints)} keypoint sets.")

    print(f"  Inference speed: {result.speed['inference']}ms, NMS speed: {result.speed['nms']}ms")

print("\nCheck the 'runs/detect/' directory for saved images/videos.")
