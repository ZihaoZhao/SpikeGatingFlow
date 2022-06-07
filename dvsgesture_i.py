import numpy as np
import os
import cv2
import h5py
import cfg
from base import DatasetBase
from visualization_utils import save_visualize, save_curve, visualize, save_vis_formatted

class DatasetGesture_i(DatasetBase):
    def __init__(self, root):
        super(DatasetGesture_i, self).__init__(root)
            
        self.input_shape = (128, 128)
        self.input_channel = self.input_shape[0] * self.input_shape[1] #?
        self.event_num = 11

        self.root = root
        self.if_save_png = False
        # self.preloaded = self.check_npy_files(self.root)
        self.preloaded = True
        self.if_dvs = True

        self.test_np_folder = os.path.join(self.root, 'test_npy')
        self.test_data_filenames = os.listdir(self.test_np_folder)
        self.test_data_filenames.sort()
        if ".DS_Store" in self.test_data_filenames:
            self.test_data_filenames.remove(".DS_Store")
        self.test_label = self.get_labels(self.root) #?

    def test_len(self):
        return len(self.test_data_filenames)


    def get_test_sample(self, i, reverse=False):
        assert i < self.test_len()
        self.test_data_filenames.sort(reverse=reverse)
        data_filename = self.test_data_filenames[i]
        np_name = os.path.join(self.test_np_folder, data_filename)
        # print(np_name)
        video = np.load(np_name)
        test_label = int(data_filename.split('_')[1])
        # class_i = int(data_filename.split('_')[2][:-4])
        # print(data_filename, video.shape)
        return video, test_label

    def get_test_data_file_name(self,i):
        return self.test_data_filenames[i]    


    def read_from_npy(self, file_name):
        video = np.load(file_name)
        return video

    def collect_data(self, dir, file_names):
        data = list()
        for data_filename in file_names:
            # print(data_filename.split(".")[0])
            save_dir = os.path.join(self.save_folder, data_filename.split(".")[0])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            f = h5py.File(os.path.join(dir, data_filename),'r')
 

            step = 1000
            video = list()
            image = np.zeros((128, 128))
            for i, addr in enumerate(f['addrs']):
                if addr[2] == 0:
                    image[addr[1]][addr[0]] = -1
                elif addr[2] == 1:
                    image[addr[1]][addr[0]] = 1
                if i % step == step - 1:
                    video.append(image)
                    image = np.zeros((128, 128))
            video = np.array(video)
            data.append(video)
            print(data_filename, len(video))

            # if self.if_save_png:
            #     for i, image in enumerate(video):
            #         vis.save_visualize(image, (128, 128), os.path.join(save_dir, str(i)+".png"))

        return data

    def collect_data_npy(self, dir, file_names):
        data = list()
        for data_filename in file_names:
            np_name = os.path.join(dir, data_filename)
            print(np_name)
            video = np.load(np_name)
            data.append(video)

            # print(video.shape)
        return data

    def save_data(self, data, dir, data_filenames):
        for i, video in enumerate(data):
            np_name = os.path.join(dir, data_filenames[i].replace('.hdf5', ''))
            np.save(np_name, video)
            print('saved in', np_name + '.npy')



    def h5pt2npy(self, root):

        test_folder = os.path.join(root, 'test')

        test_np_folder = os.path.join(root, 'test_npy')

        if not os.path.exists(test_np_folder):
            os.mkdir(test_np_folder)

        if self.preloaded == False:
            test_data_filenames = os.listdir(test_folder)
            test_data_filenames.sort()
            test_data = self.collect_data(test_folder, test_data_filenames)
            self.save_data(test_data, test_np_folder, test_data_filenames)  
            test_data_filenames = os.listdir(test_np_folder)
            test_data_filenames.sort()
            test_np_data = self.collect_data_npy(test_np_folder, test_data_filenames)
        else:
            test_data_filenames = os.listdir(test_np_folder)
            test_data_filenames.sort()
            test_np_data = self.collect_data_npy(test_np_folder, test_data_filenames)

        return  test_np_data, test_data_filenames 


    def get_labels(self, root):
        test_folder = os.path.join(root, 'test_npy')
        test_data_filenames = os.listdir(test_folder)
        test_data_filenames.sort()
        test_label = list()

        for t in test_data_filenames:
            if '.npy' in t:
                test_label.append(int(t.split('_')[1]))

        return  test_label


    def check_npy_files(self, root):
        test_folder = os.path.join(root, 'test')
        test_np_folder = os.path.join(root, 'test_npy')
        if len(os.listdir(test_folder)) != len(os.listdir(test_np_folder)):
            return False
        else:
            return True

    def dataconvert(self, event_number):
        self.train_dataset = np.full((80,128,128,event_number),0)
        for i in range(0,event_number):
            sample = self.train_np_data[i]
            self.train_dataset[:,:,:,i] =  sample[0:80,:,:]  


                  