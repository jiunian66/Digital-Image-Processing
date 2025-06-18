# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import json
from datetime import datetime

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB mcdax file size

# 创建必要的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 全局变量
model = None
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

# 营养信息数据库
nutrition_info = {
    'apple': {'calories': 52, 'vitamin_c': 4.6, 'fiber': 2.4, 'benefits': '富含纤维，有助消化'},
    'banana': {'calories': 89, 'vitamin_c': 8.7, 'fiber': 2.6, 'benefits': '富含钾，补充能量'},
    'beetroot': {'calories': 43, 'vitamin_c': 4.9, 'fiber': 2.8, 'benefits': '富含叶酸，降血压'},
    'bell pepper': {'calories': 31, 'vitamin_c': 127.7, 'fiber': 2.5, 'benefits': '维C含量极高'},
    'cabbage': {'calories': 25, 'vitamin_c': 36.6, 'fiber': 2.5, 'benefits': '抗氧化，防癌'},
    'carrot': {'calories': 41, 'vitamin_c': 5.9, 'fiber': 2.8, 'benefits': '富含胡萝卜素'},
    'corn': {'calories': 86, 'vitamin_c': 6.8, 'fiber': 2.7, 'benefits': '提供能量，含叶黄素'},
    'cucumber': {'calories': 16, 'vitamin_c': 2.8, 'fiber': 0.5, 'benefits': '补水利尿'},
    'eggplant': {'calories': 25, 'vitamin_c': 2.2, 'fiber': 3.0, 'benefits': '降胆固醇'},
    'garlic': {'calories': 149, 'vitamin_c': 31.2, 'fiber': 2.1, 'benefits': '抗菌消炎'},
    'ginger': {'calories': 80, 'vitamin_c': 5.0, 'fiber': 2.0, 'benefits': '暖胃止呕'},
    'grapes': {'calories': 62, 'vitamin_c': 10.8, 'fiber': 0.9, 'benefits': '抗氧化，护心脏'},
    'lemon': {'calories': 29, 'vitamin_c': 53.0, 'fiber': 2.8, 'benefits': '富含维C，美白'},
    'lettuce': {'calories': 15, 'vitamin_c': 9.2, 'fiber': 1.3, 'benefits': '低热量，助减肥'},
    'mango': {'calories': 60, 'vitamin_c': 36.4, 'fiber': 1.6, 'benefits': '富含维A，护眼'},
    'onion': {'calories': 40, 'vitamin_c': 7.4, 'fiber': 1.7, 'benefits': '杀菌，降血糖'},
    'orange': {'calories': 47, 'vitamin_c': 53.2, 'fiber': 2.4, 'benefits': '维C丰富，增免疫'},
    'potato': {'calories': 77, 'vitamin_c': 19.7, 'fiber': 2.2, 'benefits': '提供淀粉，饱腹'},
    'tomato': {'calories': 18, 'vitamin_c': 13.7, 'fiber': 1.2, 'benefits': '富含番茄红素'},
    'watermelon': {'calories': 30, 'vitamin_c': 8.1, 'fiber': 0.4, 'benefits': '清热解渴，利尿'}
}


def load_model():
    """加载训练好的模型"""
    global model
    try:
        # 尝试加载不同位置的模型
        model_paths = [
            'vegetable_fruit_final.pt',
            'runs/detect/vegetable_fruit_detection3/weights/best.pt',
            'runs/detect/vegetable_fruit_detection/weights/best.pt',
            'yolov8s.pt'  # 备用预训练模型
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"✅ 成功加载模型: {model_path}")
                return True

        print("❌ 找不到训练好的模型，使用预训练模型")
        model = YOLO('../yolov8s.pt')
        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False


def allowed_file(filename):
    """检查文件类型"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """预测图像中的果蔬"""
    global model

    if model is None:
        return None, "模型未加载"

    try:
        # 预测
        results = model(image_path)

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            detections = []

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                box = boxes.xyxy[i].tolist()

                if class_id < len(class_names):
                    class_name = class_names[class_id]

                    # 获取营养信息
                    nutrition = nutrition_info.get(class_name, {
                        'calories': 0, 'vitamin_c': 0, 'fiber': 0, 'benefits': '暂无信息'
                    })

                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': box,
                        'nutrition': nutrition
                    })

            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections, None
        else:
            return [], "未检测到果蔬"

    except Exception as e:
        return None, f"预测失败: {str(e)}"


def draw_results(image_path, detections):
    """在图像上绘制检测结果"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        # 复制图像用于绘制
        result_image = image.copy()

        for detection in detections:
            box = detection['box']
            class_name = detection['class']
            confidence = detection['confidence']

            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # 绘制标签背景
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), (0, 255, 0), -1)

            # 绘制标签文字
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return result_image

    except Exception as e:
        print(f"绘制结果失败: {e}")
        return None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和预测"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 预测
            detections, error = predict_image(filepath)

            if error:
                return jsonify({'error': error}), 500

            # 绘制结果
            result_image = draw_results(filepath, detections)
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

            if result_image is not None:
                cv2.imwrite(result_path, result_image)

            # 转换为base64用于前端显示
            def image_to_base64(img_path):
                with open(img_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()

            original_b64 = image_to_base64(filepath)
            result_b64 = image_to_base64(result_path) if result_image is not None else original_b64

            return jsonify({
                'success': True,
                'detections': detections,
                'original_image': original_b64,
                'result_image': result_b64,
                'filename': filename
            })

        return jsonify({'error': '不支持的文件类型'}), 400

    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/nutrition/<fruit_name>')
def get_nutrition(fruit_name):
    """获取营养信息"""
    nutrition = nutrition_info.get(fruit_name.lower(), {})
    return jsonify(nutrition)


if __name__ == '__main__':
    print("🚀 启动果蔬识别Web应用...")

    # 加载模型
    if load_model():
        print("✅ 模型加载成功")
    else:
        print("❌ 模型加载失败")

    print("🌐 访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=500)