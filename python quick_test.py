# quick_test.py
from ultralytics import YOLO
from pathlib import Path
import random


def quick_test():
    """快速测试训练好的模型"""

    # 加载模型
    model = YOLO('vegetable_fruit_final.pt')

    # 从训练数据中随机选择图像测试
    train_path = Path("E:/data/train")

    if train_path.exists():
        # 随机选择一个类别
        classes = [d for d in train_path.iterdir() if d.is_dir()]
        if classes:
            random_class = random.choice(classes)
            images = list(random_class.glob("*.jpg")) + list(random_class.glob("*.png"))

            if images:
                random_image = random.choice(images)
                print(f"测试图像: {random_image}")
                print(f"真实类别: {random_class.name}")

                # 预测
                results = model(str(random_image))

                # 显示结果
                if len(results) > 0:
                    results[0].show()  # 显示预测结果
                    results[0].save()  # 保存结果图像
                    print("✅ 预测完成，结果已显示和保存")
                else:
                    print("❌ 未检测到对象")
            else:
                print("❌ 找不到图像文件")
        else:
            print("❌ 找不到类别文件夹")
    else:
        print("❌ 训练数据路径不存在")


if __name__ == "__main__":
    quick_test()