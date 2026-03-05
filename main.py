import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# 1. 初始化参数
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 模型保存路径
model_path = "face_recognizer_model.yml"
# 标签保存路径
label_map_path = "label_map.npy"

faces = []
labels = []
label_map = {}
current_label = 0

# 支持读取中文路径的图片
def imread_train_data(path):
    """
    读取中文路径的图片并返回灰度图
    :param path: 图片路径
    :return: 灰度图像数组，如果读取失败返回None
    """
    try:
        img = Image.open(path).convert('L')
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"读取图片失败 {path}: {e}")
        return None

def load_training_data(data_dir):
    global current_label, label_map
    # 清空原有数据
    faces.clear()
    labels.clear()
    label_map.clear()
    current_label = 0
    
    if not os.path.exists(data_dir):
        return False
    
    # 遍历data目录下的所有子文件夹（每个文件夹对应一个人）
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        print(f"正在加载 {person_name} 的人脸数据...")
        
        # 为每个人分配唯一标签
        label_map[current_label] = person_name
        # 遍历该人的所有图片
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = imread_train_data(img_path)
            if img is None:
                continue
            
            # 检测图片中的人脸
            face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in face_rects:
                face_roi = img[y:y+h, x:x+w]
                faces.append(face_roi)
                labels.append(current_label)
        
        current_label += 1
    
    print(f"总共加载到 {len(faces)} 个人脸样本")
    return len(faces) > 0

def draw_text(img, text, position, font_size, color):
    """
    在OpenCV图像上绘制中文
    :param img: OpenCV格式的图像 (numpy array)
    :param text: 要绘制的文本
    :param position: 文本位置 (x, y)
    :param font_size: 字体大小
    :param color: 文字颜色 (B, G, R) 元组
    :return: 绘制了中文的OpenCV图像
    """
    # 将OpenCV的BGR图像转换为RGB的PIL图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)

    # 加载中文字体 (请根据系统实际路径修改) Windows系统通常可以在 C:\Windows\Fonts\ 下找到
    font_path = "C:/Windows/Fonts/simhei.ttf"
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # 绘制文字 (注意PIL使用RGB颜色)
    draw.text(position, text, font=font, fill=color[::-1]) # color[::-1] 将BGR转为RGB

    # 将PIL图像转换回OpenCV的BGR图像
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def train_and_save_model():
    if not load_training_data("data"):
        print("未加载到训练数据，无法训练模型！")
        return False
    
    # 训练模型
    recognizer.train(faces, np.array(labels))
    # 保存模型到本地
    recognizer.save(model_path)
    # 保存标签映射（字典）到本地
    np.save(label_map_path, label_map)
    print(f"模型已保存到：{model_path}")
    print(f"标签映射已保存到：{label_map_path}")
    return True

def load_saved_model():
    global label_map
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        return False
    
    # 加载模型
    recognizer.read(model_path)
    # 加载标签映射
    label_map = np.load(label_map_path, allow_pickle=True).item()
    print(f"已加载保存的模型：{model_path}")
    return True

def real_time_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return
    
    print("人脸识别已启动，按 'q' 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in face_rects:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_roi)
            
            # 确定显示的姓名和颜色
            if confidence < 80:
                name = label_map.get(label, "Unknown")
                display_text = name
                color = (0, 255, 0)  # 识别成功显示绿色
            else:
                display_text = "Unknown"
                color = (0, 0, 255)  # 识别失败显示红色
            
            # 绘制人脸矩形框
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # 计算右下角文字坐标
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            text_x = x + w - text_width - 5
            text_y = y + h - 5
            
            # 绘制文字
            frame = draw_text(frame, display_text, (text_x, text_y), 30, color)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"已创建 {data_dir} 文件夹，请在其中创建子文件夹（命名为姓名），并放入该人的人脸图片")
        exit()
    
    if not load_saved_model():
        print("未找到已保存的模型，开始训练新模型...")
        if train_and_save_model():
            print("模型训练完成，准备启动识别...")
        else:
            print("模型训练失败，无法启动识别！")
            exit()

    real_time_recognition()