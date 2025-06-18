# train_vegetable_fruit_model_gpu.py
import os
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
import random
import torch


def check_device():
    """检查可用设备"""
    print("设备检查:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return 0  # 使用第一个GPU
    else:
        print("将使用CPU进行训练")
        return 'cpu'


def prepare_data_structure():
    """准备数据结构，将分类数据转换为YOLO检测格式"""

    base_path = Path("E:/data")
    train_path = base_path / "train"

    if not train_path.exists():
        print(f"训练数据路径 {train_path} 不存在")
        return None

    # 创建YOLO格式的目录结构
    yolo_path = base_path / "yolo_format"
    (yolo_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 35个类别的果蔬名称
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

    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    all_images = []

    # 收集所有图像文件
    print("收集图像文件...")
    for class_folder in train_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            if class_name in class_to_id:
                class_id = class_to_id[class_name]

                # 获取该类别的所有图像
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                for ext in image_extensions:
                    for img_file in class_folder.glob(ext):
                        all_images.append((img_file, class_id, class_name))

    if not all_images:
        print("未找到任何图像文件")
        return None

    print(f"找到 {len(all_images)} 张图像")

    # 随机打乱并分割数据集
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))  # 80%训练，20%验证

    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    # 处理训练集
    print("处理训练集...")
    for i, (img_path, class_id, class_name) in enumerate(train_images):
        if i % 100 == 0:
            print(f"已处理 {i}/{len(train_images)} 张训练图像")

        # 复制图像
        new_img_name = f"{class_name}_{img_path.stem}_{i}.jpg"
        new_img_path = yolo_path / "images" / "train" / new_img_name
        try:
            shutil.copy2(img_path, new_img_path)
        except Exception as e:
            print(f"复制图像失败 {img_path}: {e}")
            continue

        # 创建YOLO格式标签
        label_path = yolo_path / "labels" / "train" / f"{new_img_name.replace('.jpg', '.txt')}"
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

    # 处理验证集
    print("处理验证集...")
    for i, (img_path, class_id, class_name) in enumerate(val_images):
        if i % 100 == 0:
            print(f"已处理 {i}/{len(val_images)} 张验证图像")

        # 复制图像
        new_img_name = f"{class_name}_{img_path.stem}_val_{i}.jpg"
        new_img_path = yolo_path / "images" / "val" / new_img_name
        try:
            shutil.copy2(img_path, new_img_path)
        except Exception as e:
            print(f"复制图像失败 {img_path}: {e}")
            continue

        # 创建YOLO格式标签
        label_path = yolo_path / "labels" / "val" / f"{new_img_name.replace('.jpg', '.txt')}"
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

    print(f"数据准备完成:")
    print(f"训练集: {len(train_images)} 张图像")
    print(f"验证集: {len(val_images)} 张图像")

    # 创建配置文件
    config = {
        'train': str(yolo_path / "images" / "train"),
        'val': str(yolo_path / "images" / "val"),
        'nc': len(class_names),
        'names': class_names
    }

    with open('vegetable_fruit_dataset.yaml', 'w') as f:
        yaml.dump(config, f)

    return 'vegetable_fruit_dataset.yaml'


def train_model():
    """训练YOLOv8模型"""
    # 检查设备
    device = check_device()

    print("开始准备数据...")
    config_path = prepare_data_structure()

    if config_path is None:
        print("数据准备失败")
        return None

    print("开始训练模型...")

    # 加载预训练模型
    try:
        model = YOLO('yolov8s.pt')  # 使用smaller版本，更适合CPU/单GPU训练
        print("成功加载预训练模型")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        return None

    # 训练参数
    train_args = {
        'data': config_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16 if device != 'cpu' else 8,  # GPU使用更大batch size
        'name': 'vegetable_fruit_detection',
        'save': True,
        'cache': False,
        'device': device,
        'workers': 4 if device != 'cpu' else 2,
        'patience': 20,
        'save_period': 10,
        'val': True,
        'plots': True,
        'cos_lr': True,
        'close_mosaic': 10
    }

    print(f"训练参数: {train_args}")

    # 开始训练
    try:
        results = model.train(**train_args)
        print("模型训练完成！")

        # 保存最终模型
        model.save('vegetable_fruit_final.pt')
        print("模型已保存为 vegetable_fruit_final.pt")

        return model
    except Exception as e:
        print(f"训练失败: {e}")
        return None


def main():
    """主函数"""
    print("果蔬识别模型训练程序")
    print("=" * 50)

    # 检查数据路径
    data_path = Path("E:/data/train")
    if not data_path.exists():
        print(f"错误: 数据路径 {data_path} 不存在")
        return

    # 显示数据集信息
    folders = [f for f in data_path.iterdir() if f.is_dir()]
    print(f"找到 {len(folders)} 个类别文件夹")

    # 开始训练
    trained_model = train_model()

    if trained_model:
        print("训练成功完成！")
        print("可以使用训练好的模型进行推理了。")
    else:
        print("训练失败，请检查错误信息。")


if __name__ == "__main__":
    main()