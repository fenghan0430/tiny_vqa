import tensorflow as tf
from keras.applications import VGG16
from keras.api._v2.keras.layers import Input, LSTM, Dense, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Multiply, Lambda, Softmax, Dropout, Embedding, Flatten, Reshape, Conv2DTranspose
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.losses import KLDivergence
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from keras.api._v2.keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences
import gc
import datetime
import cv2 
import os
import json
import numpy as np
import threading
import multiprocessing
from load_foodnet import load_floodnet

def set_gpu_memory_mode():
    """设置gpu不占满显存"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Error setting memory growth: ", e)
set_gpu_memory_mode()
    
def preprocess_data(images, questions, answers, image_labels, num_answers=None):
    '''数据预处理'''
    # 图像预处理
    # image_generator = ImageDataGenerator(rescale=1./255)
    # images = np.stack(images, axis=0) # (图片数量,h,w,通道)
    # image_data = image_generator.flow(images, batch_size=len(images), shuffle=False)
    images = np.stack(images, axis=0) # (图片数量,h,w,通道)
    min_value = np.min(images)
    max_value = np.max(images)
    image_data = (images - min_value) / (max_value - min_value)
    
    # 问题预处理
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    question_data = pad_sequences(sequences, maxlen=len(questions))
    
    # 答案预处理
    # 将布尔值转换为字符串
    answers_str = []
    for item in answers:
        answers_str.append(str(item))
    # 构建标签到整数的映射字典
    label_set = set(answers_str)
    label_map = {}
    for i, label in enumerate(label_set):
        label_map[label] = i
    # 将字符串标签转换为整数编码
    answers_encoded = []
    for label in answers_str:
        answers_encoded.append(label_map[label])

    answer_data = tf.keras.utils.to_categorical(answers_encoded, num_classes=len(label_map))
    
    # 图像标签预处理
    processed_image_labels = []    
    for img in image_labels:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, binary_image = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)        
        # 归一化
        normalized_image = binary_image / 255.0
        processed_image_labels.append(normalized_image)
    processed_image_labels = np.array(processed_image_labels)
    
    # return tf.data.Dataset.from_tensor_slices((image_data, question_data, answer_data)), len(label_map)
    return image_data, question_data, answer_data, processed_image_labels, len(label_map)

# MFB融合块
def mfb_fusion(image_features, question_features, dim_k, dim_v):
    '''MFB融合块'''
    
    # 扩展阶段
    
    # image_features_Flattened = Flatten()(image_features)
    
    image_features = Dense(dim_k)(image_features) 
    question_features = Dense(dim_k)(question_features)
    fused = Multiply()([image_features, question_features])
    fused = Dropout(0.1)(fused)
    
    # 压缩阶段
    fused = Dense(dim_v, use_bias=False)(fused) 
    fused = Lambda(lambda x: tf.reduce_sum(x, axis=1))(fused)
    fused = Lambda(lambda x: tf.nn.l2_normalize(x, dim=1))(fused)
    return fused

# 基准VQA模型
def baseline_vqa_model(vocab_size, num_answers, dim_k, dim_v):
    '''基准VQA模型'''
    # 图像输入
    image_input = Input(shape=(224, 224, 3))
    # 使用VGG16提取图像特征
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    image_features = vgg(image_input)
    
    # 问题输入和LSTM
    question_input = Input(shape=(None,), dtype='int32')
    embedding = Embedding(vocab_size, 300, mask_zero=True)(question_input)
    question_features = LSTM(1024, return_sequences=True)(embedding)
    question_features = LSTM(1024)(question_features)
    
    # MFB融合
    fused = mfb_fusion(image_features, question_features, dim_k, dim_v)
    
    # 计算注意力权重
    attention_weights = Softmax()(fused)
    # 注意力匹配损失
    attention_weights_loss = Reshape((64, 112))(attention_weights)
    
    # 应用注意力权重
    # 将形状为 (7, 1024) 的张量扩展为 (7, 7, 1024)
    expanded_question_features = tf.expand_dims(attention_weights, axis=1)
    expanded_question_features = tf.tile(expanded_question_features, [1, 7, 1, 1])
    # 使用全连接层调整问题特征张量的维度
    adjusted_question_features = Dense(512)(expanded_question_features)
    weighted_image_features = Multiply()([image_features, adjusted_question_features])
    weighted_image_features = Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]))(weighted_image_features)
    
    # 连接加权图像特征和问题特征
    combined = tf.keras.layers.concatenate([weighted_image_features, question_features])
    output = Dense(num_answers, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, question_input], outputs=[output, attention_weights_loss])
    # model = Model(inputs=[image_input, question_input], outputs=output)
    return model

# 紧凑的TinyVQA模型
def compact_vqa_model(vocab_size, num_answers):
    '''紧凑的TinyVQA模型'''
    # 图像输入
    image_input = Input(shape=(224, 224, 3))
    # 紧凑的CNN提取图像特征
    x = Conv2D(32, 3, activation='relu', padding='same')(image_input)
    x = MaxPooling2D()(x)
    x = DepthwiseConv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = DepthwiseConv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    image_features = Flatten()(x)
    
    # 问题输入和LSTM
    question_input = Input(shape=(None,), dtype='int32')
    embedding = tf.keras.layers.Embedding(vocab_size, 300, mask_zero=True)(question_input)
    question_features = LSTM(32)(embedding)
    
    # 连接图像和问题特征
    combined = tf.keras.layers.concatenate([image_features, question_features])
    combined = Dense(64, activation='relu')(combined)
    combined = Dense(128, activation='relu')(combined)
    output = Dense(num_answers, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, question_input], outputs=output)
    return model

# 知识蒸馏损失
def kd_loss(y_true, y_pred, temp=1.0):
    '''知识蒸馏损失'''
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 应用温度缩放
    y_true = tf.nn.softmax(y_true / temp)
    y_pred = tf.nn.softmax(y_pred / temp)
    
    # 计算KL散度损失
    loss = KLDivergence()(y_true, y_pred)
    return loss


# 示例用法
vocab_size = 10000
dim_k = 1024
dim_v = 1024
max_seq_length = 20
batch_size = 64
# baseline_model = baseline_vqa_model(vocab_size, 41, dim_k, dim_v)
# baseline_model.summary()
temp = load_floodnet("floodnet_dataset")
images, questions, answers, image_labels = temp.load_data()
image_data, question_data, answer_data, image_labels_data, num_answers = preprocess_data(images, questions, answers, image_labels)
dataset = tf.data.Dataset.from_tensor_slices((image_data, question_data, answer_data, image_labels_data))
def process_data(image, question, answer, label):
    """用于解包元组并提供图像和问题"""
    return (image, question), (answer, label)
    # return {"input_1":image, "input_3":question}, answer
# 应用处理函数到数据集
dataset = dataset.map(process_data)
# 批量化数据集
dataset = dataset.batch(batch_size)

# # 训练基准VQA模型
baseline_model = baseline_vqa_model(vocab_size, num_answers, dim_k, dim_v)
baseline_model.compile(optimizer='adam', loss=['categorical_crossentropy', tf.keras.losses.MeanSquaredError()], metrics=['accuracy'])
# baseline_model.summary()
baseline_model.fit(dataset, epochs=10)
del dataset
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
weights_save_path = f"baseline2-{current_time}.h5"
baseline_model.save_weights(weights_save_path)
print(f"Weights saved to {weights_save_path}")
    