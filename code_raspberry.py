import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
import time
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db
from appJar import gui
import threading
from PIL import Image
from datetime import datetime

# Đặt chế độ GPIO
GPIO.setmode(GPIO.BCM)

# Đặt chân GPIO
PIN_SENSOR = 14
PIN_MOTOR = 23
SERVO_PIN_1 = 27
SERVO_PIN_2 = 17
SERVO_PIN_3 = 22

# Đặt chân GPIO là INPUT và OUTPUT
GPIO.setup(PIN_SENSOR, GPIO.IN)
GPIO.setup(PIN_MOTOR, GPIO.OUT)
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)
GPIO.setup(SERVO_PIN_3, GPIO.OUT)

# Tạo đối tượng PWM
pwm1 = GPIO.PWM(SERVO_PIN_1, 50) # GPIO 27 for PWM with 50Hz
pwm2 = GPIO.PWM(SERVO_PIN_2, 50) # GPIO 17 for PWM with 50Hz
pwm3 = GPIO.PWM(SERVO_PIN_3, 50) # GPIO 22 for PWM with 50Hz

# Khởi động PWM:
pwm1.start(12)
pwm2.start(2) 
pwm3.start(8.1) 

# Tải mô hình đã huấn luyện
model = load_model('/home/pi/Desktop/TUAN12/model_fine_tuned.h5', compile=False)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Khởi tạo biến trạng thái
running = False
captured = False

# Đường dẫn đến tệp JSON bạn đã tải về từ Firebase
firebase_cred_path = "/home/pi/Desktop/Firebase/banana-classification-33254-firebase-adminsdk-h6bx6-feb0744f5a.json"

# Khởi tạo Firebase với thông tin đăng nhập từ tệp JSON
cred = credentials.Certificate(firebase_cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://banana-classification-33254-default-rtdb.firebaseio.com/'
})

# Tạo một tham chiếu đến cơ sở dữ liệu
ref = db.reference('banana_counts')

# Chương trình con cập nhật dữ liệu lên firebase
def update_banana_count(banana_type):
    current_count = ref.child(banana_type).get() if ref.child(banana_type).get() else 0
    updated_count = current_count + 1
    ref.update({banana_type: updated_count})

# Chương trình con cập nhật giao diện người dùng
def update_user_interface(banana_type, frame):
    app.queueFunction(app.setLabel, "bananaLabel", banana_type)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
    image = image.resize((640, 480))  
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  
    filename = f"/home/pi/Desktop/du lieu/picture/temp_{timestamp}.gif"  
    image.save(filename)
    app.queueFunction(app.setImage, "bananaImage", filename)

# Chương trình con điều khiển servo 1
def control_servo1():
    pwm1.ChangeDutyCycle(8) 
    time.sleep(2.7)
    pwm1.ChangeDutyCycle(12)

# Chương trình con điều khiển servo 2
def control_servo2():
    pwm2.ChangeDutyCycle(4.4) 
    time.sleep(2.8)
    pwm2.ChangeDutyCycle(2)

# Chương trình con điều khiển servo 3
def control_servo3():
    pwm3.ChangeDutyCycle(12) 
    time.sleep(3.5)
    pwm3.ChangeDutyCycle(8.1)

# Chương trình con xử lý và dự đoán ảnh
def predict_image(frame):
    # Tiền xử lý ảnh
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(frame, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Dự đoán
    prediction = model.predict(image)
    max_value = np.max(prediction)
    prediction = np.argmax(prediction, axis=1)
    
    return prediction, max_value

# Chương trình con chụp ảnh
def capture_image():
    ret, frame = cap.read()
    return frame

# Chương trình con so sánh và gán nhãn loại chuối
def classify_and_label_banana(frame):
    prediction, max_value = predict_image(frame)
    if max_value < 0.93:  
        return "LoaiKhac"
    elif prediction[0] == 0:
        return "ChuoiLun"
    elif prediction[0] == 1:
        return "ChuoiSu"
    elif prediction[0] == 2:
        return "ChuoiCau"

# Chương trình chính
def run_loop():
    global running, captured
    while running:
        sensor_value = GPIO.input(PIN_SENSOR)
        if sensor_value == 0:
            if not captured:
                time.sleep(1)
                GPIO.output(PIN_MOTOR,GPIO.LOW)
                time.sleep(1)
                frame = capture_image()
                 GPIO.output(PIN_MOTOR,GPIO.HIGH)
                banana_type = classify_and_label_banana(frame)
                if banana_type == "LoaiKhac":
                    print("Loại khác")
                    update_banana_count("LoaiKhac")
                    update_user_interface("LoaiKhac", frame)
                elif banana_type == "ChuoiLun":
                    print("Chuối lùn")
                    update_banana_count("ChuoiLun")
                    update_user_interface("ChuoiLun", frame)
                    control_servo2()
                elif banana_type == "ChuoiSu":
                    print("Chuối sứ")
                    update_banana_count("ChuoiSu")
                    update_user_interface("ChuoiSu", frame)
                    time.sleep(4)
                    control_servo3()
                elif banana_type == "ChuoiCau":
                    print("Chuối cau")
                    update_banana_count("ChuoiCau")
                    update_user_interface("ChuoiCau", frame)
                    control_servo1()
                captured = True
        else:
            captured = False
        time.sleep(0.1)

# Chương trình nút nhấn Start
def start(btn):
    global running, captured
    running = True
    captured = False
    GPIO.output(PIN_MOTOR, GPIO.LOW)
    threading.Thread(target=run_loop).start()
    app.setButtonBg("Start", "green")
    app.setButtonBg("Stop", "white")

# Chương trình nút nhấn Stop
def stop(btn):
    global running, captured
    running = False
    captured = False
    GPIO.output(PIN_MOTOR, GPIO.HIGH)
    app.setButtonBg("Stop", "yellow")
    app.setButtonBg("Start", "white")

# Chương trình nút nhấn Exit
def exit(btn):
    app.stop()

try:
    app = gui("Conveyor Control")
    app.setBg("white")
    app.setFont(18)
    app.setSize("FullScreen")  
    app.addImage("bananaImage", "/home/pi/Desktop/du lieu/picture/empty.gif", 0, 0, 2)  
    app.setImageWidth("bananaImage", 640)  
    app.setImageHeight("bananaImage", 480)  
    app.startLabelFrame("Class", 1, 0, 2)  
    app.setStretch("both") 
    app.setSticky("nesw")  
    app.addLabel("bananaLabel", "")  
    app.stopLabelFrame()  
    app.addButton("Start", start, 2, 0) 
    app.addButton("Stop", stop, 2, 1)  
    app.addButton("Exit", exit, 3, 0, 2)  
    app.setButtonWidth("Start", 20)  
    app.setButtonHeight("Start", 2)  
    app.setButtonWidth("Stop", 20)  
    app.setButtonHeight("Stop", 2)  
    app.setButtonWidth("Exit", 20)  
    app.setButtonHeight("Exit", 2)  
    app.go()
except KeyboardInterrupt:
    GPIO.cleanup()
    pwm1.stop()
    pwm2.stop()
    pwm3.stop()
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f"Error: {str(e)}")
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()