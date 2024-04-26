import tensorflow as tf
from keras.applications import VGG16
from keras.api._v2.keras.layers import Input, LSTM, Dense, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Multiply, Lambda, Softmax, Dropout, Embedding, Flatten, Reshape, Conv2DTranspose
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.losses import KLDivergence

class vqa_model:
    def __init__(
        self, 
        num_answers, 
        vocab_size=10000, 
        dim_k=1024, 
        dim_v=1024
        ):
        
        self.vocab_size = vocab_size
        self.num_answers = num_answers
        self.dim_k = dim_k
        self.dim_v = dim_v
    
    def mfb_fusion(self, image_features, question_features):
        '''MFB融合块'''
        
        # 扩展阶段
        
        # image_features_Flattened = Flatten()(image_features)
        
        image_features = Dense(self.dim_k)(image_features) 
        question_features = Dense(self.dim_k)(question_features)
        fused = Multiply()([image_features, question_features])
        fused = Dropout(0.1)(fused)
        
        # 压缩阶段
        fused = Dense(self.dim_v, use_bias=False)(fused) 
        fused = Lambda(lambda x: tf.reduce_sum(x, axis=1))(fused)
        fused = Lambda(lambda x: tf.nn.l2_normalize(x, dim=1))(fused)
        return fused

    def baseline_vqa_model(self) -> Model:
        '''基准VQA模型'''
        # 图像输入
        image_input = Input(shape=(224, 224, 3))

        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        image_features = vgg(image_input)
        
        # 问题输入
        question_input = Input(shape=(None,), dtype='int32')
        embedding = Embedding(self.vocab_size, 300, mask_zero=True)(question_input)
        question_features = LSTM(1024, return_sequences=True)(embedding)
        question_features = LSTM(1024)(question_features)
        
        # MFB融合
        fused = self.mfb_fusion(image_features, question_features)
        
        # 计算注意力权重
        attention_weights_1 = Softmax()(fused)
        attention_weights_Flattened = Flatten()(attention_weights_1)
        attention_weights_loss = Dense(7168)(attention_weights_Flattened)
        
        # 应用注意力权重
        # 将形状为 (7, 1024) 的张量扩展为 (7, 7, 1024)
        attention_weights = Reshape((7,1024))(attention_weights_loss)
        expanded_question_features = tf.expand_dims(attention_weights, axis=1)
        expanded_question_features = tf.tile(expanded_question_features, [1, 7, 1, 1])
        # 使用全连接层调整问题特征张量的维度
        adjusted_question_features = Dense(512)(expanded_question_features)
        weighted_image_features = Multiply()([image_features, adjusted_question_features])
        weighted_image_features = Lambda(lambda x: tf.reduce_sum(x, axis=[1, 2]))(weighted_image_features)
        
        # 连接加权图像特征和问题特征
        combined = tf.keras.layers.concatenate([weighted_image_features, question_features])
        output = Dense(self.num_answers, activation='softmax')(combined)
        
        model = Model(inputs=[image_input, question_input], outputs=[output, attention_weights_loss])
        # model = Model(inputs=[image_input, question_input], outputs=output)
        return model

    # 紧凑的TinyVQA模型
    def compact_vqa_model(self) -> Model:
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
        embedding = tf.keras.layers.Embedding(self.vocab_size, 300, mask_zero=True)(question_input)
        question_features = LSTM(32)(embedding)
        
        # 连接图像和问题特征
        combined = tf.keras.layers.concatenate([image_features, question_features])
        combined = Dense(64, activation='relu')(combined)
        combined = Dense(128, activation='relu')(combined)
        output = Dense(self.num_answers, activation='softmax')(combined)
        
        model = Model(inputs=[image_input, question_input], outputs=output)
        return model
    
        # 知识蒸馏损失
    def kd_loss(self, y_true, y_pred, temp=1.0):
        '''知识蒸馏损失'''
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 应用温度缩放
        y_true = tf.nn.softmax(y_true / temp)
        y_pred = tf.nn.softmax(y_pred / temp)
        
        # 计算KL散度损失
        loss = KLDivergence()(y_true, y_pred)
        return loss

# if __name__ == "__main__":
#     import numpy as np
#     vocab_size = 10000
#     dim_k = 1024
#     dim_v = 1024
#     max_seq_length = 20
#     batch_size = 64
#     baseline_model = baseline_vqa_model(vocab_size, 41, dim_k, dim_v)
#     baseline_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
#     baseline_model.summary()
#     # 创建测试数据
#     test_image = np.random.random((1, 224, 224, 3))  # 创建一个随机图像数据
#     test_vector = np.random.random((1, 1024))  # 创建一个随机向量数据
#     predictions = baseline_model.predict([test_image, test_vector])
#     # 检查输出
#     print(predictions[0].shape)
#     print(predictions[1].shape)