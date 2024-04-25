import json
import multiprocessing
import os
import threading
import cv2

class load_floodnet():
    def __init__(self, dir):
        self.dir = dir # 数据集路径
        
        self.resized_images_dict = multiprocessing.Manager().dict() # 多线程字典
        self.resized_images_label_dict = multiprocessing.Manager().dict()
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
        '''
        以(224,224,3)读取图像和图片掩码,
        存储在self.resized_images_dict和self.resized_images_label_dict中
        '''
        for path in image_paths:
            _, file_name = os.path.split(path)
            if 'lab' in file_name:
                continue
            tmp_path, tmp_filename = os.path.split(path)
            new_filename = tmp_filename.split('.')[0] + '_lab.png'
            lab_path = os.path.join(tmp_path, new_filename)
            image_lab = cv2.imread(lab_path)
            image = cv2.imread(path)
            resized_image = cv2.resize(image, (224, 224))
            resized_image_lab = cv2.resize(image_lab, (224, 224))
            with self.lock:
                # 不同的字典使用同一个图片名，以确保掩码和图片对应
                # 并方便后续处理
                self.resized_images_dict[file_name] = resized_image
                self.resized_images_label_dict[file_name] = resized_image_lab
    
    def process_images_parallel(self):
        '''多线程加载图片和图片掩码'''
        image_paths = self.get_all_filenames(os.path.join(self.dir, "Images", "Train_Image"))
        num_cores = multiprocessing.cpu_count()
        chunk_size = len(image_paths) // num_cores
        chunks = [image_paths[i:i+chunk_size] for i in range(0, len(image_paths), chunk_size)]

        processes = []
        for i, chunk in enumerate(chunks):
            process = multiprocessing.Process(target=self.resize_and_store_images, args=(chunk,), name=f"Process-{i+1}")
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        return self.resized_images_dict, self.resized_images_label_dict
    
    def load_data(self):
        '''
        加载数据,
        返回:图片列表,问题列表,答案列表,图像掩码列表
        '''
        self.resized_images_dict = self.process_images_parallel()
        
        # 从json文件中读取问题和标签, 并将标签和数据对应 
        with open(os.path.join(self.dir, "Questions", "Training Question.json"), 'r') as f:
            data = json.load(f)

        image_id_question_list=[]

        for key, value in data.items():
            image_id = value["Image_ID"]
            if image_id in self.resized_images_dict:
                image = self.resized_images_dict[image_id]
                image_lab = self.resized_images_label_dict[image_id]
            else:
                image = None
                image_lab = None
            image_id_question_list.append([image_id, value["Question"], value["Ground_Truth"], image, image_lab])
            
        return [item[3] for item in image_id_question_list], [item[1] for item in image_id_question_list], [item[2] for item in image_id_question_list], [item[4] for item in image_id_question_list]

# if __name__ == "__main__":
#     dir = "floodnet_dataset"
#     test = load_floodnet(dir)
#     images, questions, answers, labels = test.load_data()
#     print(len(images), len(questions), len(answers), len(labels))