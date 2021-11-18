import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from PIL import Image


class Face_lib(object):
    def __init__(self, face_lib_path, face_name):
        self.face_lib_path = face_lib_path
        self.face_name = face_name
        pass

    def create_file(self):
        # 先遍历所有子文件 确定从哪开始标号
        sub_file_counter = 0
        name_list = []
        for sub_file in os.listdir(self.face_lib_path):
            name_list.append(sub_file.split('_')[0])
            sub_file_counter += 1

        name_list.sort()

        if name_list == None:
            os.mkdir(self.face_lib_path + '0_' + self.face_name)
        else:
            os.mkdir(self.face_lib_path + str(sub_file_counter) + '_' + self.face_name)

    def save_face_img(self, img):
        self.create_file()
        img.save(self.face_lib_path + self.face_name + '/')


class Create_Dataset(object):
    def __init__(self, train_data_path, face_lib_path):
        self.train_data_path = train_data_path
        self.face_lib_path = face_lib_path

    def clear_dataset(self):
        for i in os.listdir(self.face_lib_path):
            os.remove(self.face_lib_path + i)

    def resize(self, output_file, width=100, height=100):
        for sub_file in os.listdir(self.face_lib_path):
            for each_face_img in os.listdir(self.face_lib_path + sub_file):
                img_path = self.face_lib_path + sub_file + '/' + each_face_img
                img = Image.open(img_path)
                img_RGB = img.convert('RGB')
                resized_img = img_RGB.resize(width, height, Image.BILINEAR)
                resized_img.save(output_file)

    def img_to_loader(self, train_img_path, batch_size, shuffle):
        img_np_list = []
        img_lb_list = []
        for train_img in os.listdir(train_img_path):
            img = Image.open(train_img_path + train_img)
            np_img = np.array(img)
            img_np_list.append(np_img)
            img_lb_list.append(train_img.split('_')[0])
        img_np_list = np.array(img_lb_list).transpose((0, 3, 1, 2))
        img_tensor = torch.from_numpy(img_np_list) / 255
        img_lb_tensor = torch.LongTensor(img_lb_list)
        data = TensorDataset(img_tensor, img_lb_tensor)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return loader
