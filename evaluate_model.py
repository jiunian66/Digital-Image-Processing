# evaluate_model.py
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def evaluate_trained_model():
    """评估训练好的模型"""
    print("果蔬识别模型评估")
    print("=" * 50)

    # 加载训练好的模型
    model_paths = [
        'vegetable_fruit_final.pt',
        'runs/detect/vegetable_fruit_detection3/weights/best.pt',
        'runs/detect/vegetable_fruit_detection3/weights/last.pt'
    ]

    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break

    if not model_path:
        print("❌ 找不到训练好的模型文件")
        return

    print(f"✅ 加载模型: {model_path}")
    model = YOLO(model_path)

    # 模型信息
    print(f"模型类型: {type(model.model)}")

    # 在验证集上评估
    print("\n📊 在验证集上评估...")
    try:
        results = model.val()
        print("验证结果:")
        print(f"  mAP50: {results.box.map50:.3f}")
        print(f"  mAP50-95: {results.box.map:.3f}")
        print(f"  精确度: {results.box.mp:.3f}")
        print(f"  召回率: {results.box.mr:.3f}")
    except Exception as e:
        print(f"验证失败: {e}")

    return model


def test_single_images(model):
    """测试单张图像"""
    print("\n🖼️ 单张图像测试")
    print("=" * 30)

    # 测试图像路径
    test_paths = [
        "E:/data/train",  # 从训练数据中随机选择
        "E:/data/test"  # 如果有测试数据
    ]

    test_images = []
    for test_path in test_paths:
        path = Path(test_path)
        if path.exists():
            # 从每个类别随机选择一张图像
            for class_folder in path.iterdir():
                if class_folder.is_dir():
                    images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
                    if images:
                        test_images.append((images[0], class_folder.name))
                        if len(test_images) >= 10:  # 限制测试图像数量
                            break
            break

    if not test_images:
        print("❌ 找不到测试图像")
        return

    print(f"找到 {len(test_images)} 张测试图像")

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

    results = []

    for i, (img_path, true_class) in enumerate(test_images):
        print(f"\n测试 {i + 1}/{len(test_images)}: {img_path.name}")
        print(f"真实类别: {true_class}")

        try:
            # 预测
            pred_results = model(str(img_path))

            if len(pred_results) > 0 and len(pred_results[0].boxes) > 0:
                # 获取最高置信度的预测
                boxes = pred_results[0].boxes
                max_conf_idx = torch.argmax(boxes.conf)

                predicted_class_id = int(boxes.cls[max_conf_idx])
                confidence = float(boxes.conf[max_conf_idx])
                predicted_class = class_names[predicted_class_id] if predicted_class_id < len(
                    class_names) else "unknown"

                print(f"预测类别: {predicted_class}")
                print(f"置信度: {confidence:.3f}")

                # 判断是否正确
                is_correct = predicted_class == true_class
                print(f"预测结果: {'✅ 正确' if is_correct else '❌ 错误'}")

                results.append({
                    'image': img_path.name,
                    'true_class': true_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'correct': is_correct
                })
            else:
                print("❌ 未检测到任何对象")
                results.append({
                    'image': img_path.name,
                    'true_class': true_class,
                    'predicted_class': 'none',
                    'confidence': 0.0,
                    'correct': False
                })

        except Exception as e:
            print(f"❌ 预测失败: {e}")

    # 统计结果
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] > 0])

        print(f"\n📈 测试统计:")
        print(f"总测试图像: {len(results)}")
        print(f"正确预测: {correct_count}")
        print(f"准确率: {accuracy:.3f} ({accuracy * 100:.1f}%)")
        print(f"平均置信度: {avg_confidence:.3f}")

        # 保存结果
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("✅ 测试结果已保存到 test_results.json")


def create_inference_demo():
    """创建推理演示脚本"""
    demo_code = '''
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
'''

    with open('inference_demo.py', 'w', encoding='utf-8') as f:
        f.write(demo_code)

    print("✅ 推理演示脚本已创建: inference_demo.py")


def main():
    """主函数"""
    print("模型后续处理选项:")
    print("1. 评估模型性能")
    print("2. 测试单张图像")
    print("3. 创建推理演示")
    print("4. 所有操作")

    choice = input("\n请选择操作 (1-4): ").strip()

    if choice in ['1', '4']:
        model = evaluate_trained_model()
    else:
        # 加载模型用于测试
        model_path = 'vegetable_fruit_final.pt'
        if Path(model_path).exists():
            model = YOLO(model_path)
        else:
            print("❌ 找不到模型文件")
            return

    if choice in ['2', '4']:
        test_single_images(model)

    if choice in ['3', '4']:
        create_inference_demo()

    print("\n🎉 操作完成！")
    print("\n📋 下一步建议:")
    print("1. 运行 inference_demo.py 测试您自己的图像")
    print("2. 优化模型参数重新训练")
    print("3. 收集更多数据扩充数据集")
    print("4. 部署模型到实际应用中")


if __name__ == "__main__":
    main()