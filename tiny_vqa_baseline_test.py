import datetime
import tensorflow as tf
from vqa_model import vqa_model
from vqa_data import vqa_data

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

if __name__ == "__main__":
    dataset_dir = "floodnet_dataset"
    batch_size = 32    
    
    _data = vqa_data(dataset_dir)
    images, questions, answers, image_labels = _data.load_data()
    image_data, question_data, answer_data, image_labels_data, num_answers = _data.preprocess_data(images, questions, answers, image_labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_data, question_data, answer_data, image_labels_data))
    def process_data(image, question, answer, label):
        """用于解包元组并提供图像和问题"""
        return (image, question), (answer, label)
    dataset = dataset.map(process_data)
    dataset = dataset.batch(batch_size)
    
    model_builder = vqa_model(num_answers)
    baseline_model = model_builder.baseline_vqa_model()
    baseline_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    baseline_model.fit(dataset, epochs=10)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    weights_save_path = f"baseline2-{current_time}.h5"
    baseline_model.save_weights(weights_save_path)
    print(f"Weights saved to {weights_save_path}")
    