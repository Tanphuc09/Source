import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
# Đường dẫn đến thư mục chứa tập dữ liệu
train_dir = '/content/drive/MyDrive/DATA_BANANA_TUAN8_THU5'
valid_dir = '/content/drive/MyDrive/VALI_BANANA_NEW'
 
# Tạo ImageDataGenerator với Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2)
 
valid_datagen = ImageDataGenerator(rescale=1./255)
 
# Tạo các đối tượng generator từ thư mục chứa ảnh
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32)
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(224, 224), batch_size=32)
 
# Tải pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False)
 
# Thêm các lớp mới
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)
 
# Tạo model mới
model = Model(inputs=base_model.input, outputs=predictions)
 
# Đóng băng các lớp của MobileNet để không cập nhật trong quá trình huấn luyện
for layer in base_model.layers:
    layer.trainable = False
 
# Biên dịch model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Tạo ModelCheckpoint
checkpoint = ModelCheckpoint('/content/drive/MyDrive/MODEL/best_model_cuoicung.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
 
# Huấn luyện model
history = model.fit(train_generator, validation_data=valid_generator, epochs=10, callbacks=[checkpoint])

