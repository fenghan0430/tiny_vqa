import tensorflow as tf
from keras.applications import VGG16
from keras.api._v2.keras.layers import Input, LSTM, Dense, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Multiply, Lambda, Softmax, Dropout, Embedding, Flatten
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.losses import KLDivergence
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from keras.api._v2.keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences

import cv2 
import os
import json
import numpy as np
import threading
import multiprocessing

class load_floodnet():
    def __init__(self):
        self.resized_images_dict = multiprocessing.Manager().dict()
        self.lock = threading.Lock()
    
    def get_all_filenames(self, directory):
        """列出文件名"""
        filenames = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                filenames.append(file_path)
        return filenames

    def resize_and_store_images(self, image_paths):
        for path in image_paths:
            image = cv2.imread(path)
            resized_image = cv2.resize(image, (224, 224))
            _, file_name = os.path.split(path)
            with self.lock:
                self.resized_images_dict[file_name] = resized_image
    
    def process_images_parallel(self):
        image_paths = self.get_all_filenames("/work/floodnet_dataset/Images/Train_Image")
        num_cores = multiprocessing.cpu_count()
        # num_cores = 8
        chunk_size = len(image_paths) // num_cores
        chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]

        processes = []
        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.resize_and_store_images, args=(chunk,), name=f"Process-{i+1}")
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
        return self.resized_images_dict
    
    def load_data(self):
        '''加载数据'''
        self.resized_images_dict = self.process_images_parallel()
        
        # 从json文件中读取问题和标签, 并将标签和数据对应 
        with open('/work/floodnet_dataset/Questions/Training Question.json', 'r') as f:
            data = json.load(f)

        image_id_question_list=[]

        for key, value in data.items():
            image_id = value["Image_ID"]
            if image_id in self.resized_images_dict:
                image = self.resized_images_dict[image_id]
            else:
                image = None
            image_id_question_list.append([image_id, value["Question"], value["Ground_Truth"], image])
        
        # 返回图片，问题和答案三个列表    
        return [item[3] for item in image_id_question_list], [item[1] for item in image_id_question_list], [item[2] for item in image_id_question_list]
    
def preprocess_data(images, questions, answers, num_answers=None):
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
    
    return tf.data.Dataset.from_tensor_slices((image_data, question_data, answer_data)), len(label_map)

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
    
    # model = Model(inputs=[image_input, question_input], outputs=[output, attention_weights])
    model = Model(inputs=[image_input, question_input], outputs=output)
    return model

# 紧凑的TinyVQA模型
def tiny_vqa_model(vocab_size, num_answers):
    '''紧凑的TinyVQA模型'''
    # 图像输入
    image_input = Input(shape=(224, 224, 3))
    # 紧凑的CNN提取图像特征
    x = Conv2D(32, 3, activation='relu')(image_input)
    x = MaxPooling2D()(x)
    x = DepthwiseConv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = DepthwiseConv2D(64, 3, activation='relu')(x)
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
# num_answers = 100
dim_k = 1024
dim_v = 1024
max_seq_length = 20

temp = load_floodnet()
images, questions, answers = temp.load_data()

dataset, num_answers = preprocess_data(images, questions, answers)
def process_data(image, question, answer):
    """用于解包元组并提供图像和问题"""
    return (image, question), answer
# 应用处理函数到数据集
dataset = dataset.map(process_data)
# 批量化数据集
dataset = dataset.batch(32)

# 训练基准VQA模型
baseline_model = baseline_vqa_model(vocab_size, num_answers, dim_k, dim_v)
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
baseline_model.fit(dataset, epochs=10)
    