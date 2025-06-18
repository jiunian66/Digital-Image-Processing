
# inference_demo.py
from ultralytics import YOLO
import cv2
from pathlib import Path

def predict_vegetable_fruit(image_path, model_path='vegetable_fruit_final.pt'):
    """预测单张图像的果蔬类别"""

    # 类别名称
    class_names = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
        'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn',
        'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes',
        'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango',
        'onion', 'orange', 'paprika', 'pear', 'peas',
        'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans',
        'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
        'watermelon'
    ]

    # 加载模型
    model = YOLO(model_path)

    # 预测
    results = model(image_path)

    # 处理结果
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes

        # 获取所有检测结果
        detections = []
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            class_name = class_names[class_id] if class_id < len(class_names) else "unknown"

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'box': boxes.xyxy[i].tolist()
            })

        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections
    else:
        return []

def visualize_prediction(image_path, model_path='vegetable_fruit_final.pt'):
    """可视化预测结果"""

    # 预测
    detections = predict_vegetable_fruit(image_path, model_path)

    # 读取图像
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 绘制检测框
    for detection in detections:
        box = detection['box']
        class_name = detection['class']
        confidence = detection['confidence']

        # 绘制矩形框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加标签
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_rgb, detections

if __name__ == "__main__":
    # 示例使用
    image_path = "test_image.jpg"  # 替换为您的图像路径

    if Path(image_path).exists():
        detections = predict_vegetable_fruit(image_path)

        print("检测结果:")
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['class']}: {det['confidence']:.3f}")
    else:
        print("请提供有效的图像路径")
