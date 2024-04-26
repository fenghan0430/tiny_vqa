import tensorflow as tf
from keras.applications import VGG16
from keras.api._v2.keras.layers import Input, LSTM, Dense, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Multiply, Lambda, Softmax, Dropout
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.losses import KLDivergence
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from keras.api._v2.keras.preprocessing.text import Tokenizer
from keras.api._v2.keras.preprocessing.sequence import pad_sequences

# MFB融合块
def mfb_fusion(image_features, question_features, dim_k, dim_v):
    # 扩展阶段
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
    # 图像输入
    image_input = Input(shape=(224, 224, 3))
    # 使用VGG16提取图像特征
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    image_features = vgg(image_input)
    
    # 问题输入和LSTM
    question_input = Input(shape=(None,), dtype='int32')
    embedding = tf.keras.layers.Embedding(vocab_size, 300, mask_zero=True)(question_input)
    question_features = LSTM(1024, return_sequences=True)(embedding)
    question_features = LSTM(1024)(question_features)
    
    # MFB融合
    fused = mfb_fusion(image_features, question_features, dim_k, dim_v)
    
    # 计算注意力权重
    attention_weights = Softmax()(fused)
    
    # 应用注意力权重
    weighted_image_features = Multiply()([image_features, attention_weights])
    weighted_image_features = Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]))(weighted_image_features)
    
    # 连接加权图像特征和问题特征 
    combined = tf.keras.layers.concatenate([weighted_image_features, question_features])
    output = Dense(num_answers, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, question_input], outputs=[output, attention_weights])
    return model

# 紧凑的TinyVQA模型
def tiny_vqa_model(vocab_size, num_answers):
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
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # 应用温度缩放
    y_true = tf.nn.softmax(y_true / temp)
    y_pred = tf.nn.softmax(y_pred / temp)
    
    # 计算KL散度损失
    loss = KLDivergence()(y_true, y_pred)
    return loss

# 数据预处理
def preprocess_data(images, questions, answers, max_seq_length, num_answers):
    # 图像预处理
    image_generator = ImageDataGenerator(rescale=1./255)
    image_data = image_generator.flow(images, batch_size=len(images), shuffle=False)
    
    # 问题预处理
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    question_data = pad_sequences(sequences, maxlen=max_seq_length)
    
    # 答案预处理
    answer_data = tf.keras.utils.to_categorical(answers, num_classes=num_answers)
    
    return image_data, question_data, answer_data

# 示例用法
vocab_size = 10000
num_answers = 100
dim_k = 1024
dim_v = 1024
max_seq_length = 20

# 加载FloodNet数据集
images = ...  # 加载FloodNet图像数据
questions = ...  # 加载FloodNet问题数据
answers = ...  # 加载FloodNet答案数据

# 数据预处理
train_images, train_questions, train_answers = preprocess_data(train_images, train_questions, train_answers, max_seq_length, num_answers)
val_images, val_questions, val_answers = preprocess_data(val_images, val_questions, val_answers, max_seq_length, num_answers)

# 训练基准VQA模型
baseline_model = baseline_vqa_model(vocab_size, num_answers, dim_k, dim_v)
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
baseline_model.fit([train_images, train_questions], train_answers, validation_data=([val_images, val_questions], val_answers), epochs=10, batch_size=32)

# 使用知识蒸馏训练紧凑的TinyVQA模型
tiny_model = tiny_vqa_model(vocab_size, num_answers)
tiny_model.compile(optimizer='adam', loss=kd_loss, metrics=['accuracy'])

# 使用基准模型的预测作为软标签
soft_labels = baseline_model.predict([train_images, train_questions])

# 在FloodNet数据集上使用软标签训练TinyVQA模型
tiny_model.fit([train_images, train_questions], soft_labels, validation_data=([val_images, val_questions], val_answers), epochs=10, batch_size=32)

# 模型量化
converter = tf.lite.TFLiteConverter.from_keras_model(tiny_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存量化后的模型
with open('tiny_vqa_model.tflite', 'wb') as f:
    f.write(tflite_model)
