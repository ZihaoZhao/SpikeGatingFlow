import numpy as np
import os
import cv2
import h5py
import shutil
import random

import cfg
from base import DatasetBase
import visualization_utils as vis

class DatasetGesture(DatasetBase):
    def __init__(self, root):
        super(DatasetGesture, self).__init__(root)
        self.save_folder = cfg.code_path + "/output/dvsframe"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.input_shape = (128, 128)
        self.input_channel = self.input_shape[0] * self.input_shape[1] #?
        self.event_num = 11
        self.root = root
        self.if_save_png = False
        self.batch_true = True
        self.if_dvs = True  
        self.train_num = 98

    def train_len(self):
        return 131#len(self.train_np_data)

    def get_train_sample(self, i):
        return self.train_np_data[i], self.train_label[i]

    def get_train_label(self, i):
        return self.train_label[i]

    def read_from_npy(self, file_name):
        video = np.load(file_name)
        return video

    def collect_save_data(self, dir, file_names):
        data = list()
        if not os.path.exists(dir.replace("train_label", "train_npy")):
            os.mkdir(dir.replace("train_label", "train_npy"))
        if not os.path.exists(dir.replace("test_label", "test_npy")):
            os.mkdir(dir.replace("test_label", "test_npy"))
            
        for data_filename in file_names:
            # print(data_filename.split(".")[0])
            # save_dir = os.path.join(self.save_folder, data_filename.split(".")[0])
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            try:
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
                print(data_filename, len(video))

                np_name = os.path.join(dir.replace("train_label", "train_npy")
                                            .replace("test_label", "test_npy")\
                    , data_filename.replace('.hdf5', ''))
                np.save(np_name, video)
                print('saved in', np_name + '.npy')

                # if self.if_save_png:
                #     for i, image in enumerate(video):
                #         vis.save_visualize(image, (128, 128), os.path.join(save_dir, str(i)+".png"))
            except:
                print(os.path.join(dir, data_filename))

    def collect_data_npy(self, dir, file_names):
        data = list()
        file_names.sort()
        for data_filename in file_names:
            np_name = os.path.join(dir, data_filename)
            video = np.load(np_name)
            data.append(video)
        return data

    def save_data(self, data, dir, data_filenames):
        for i, video in enumerate(data):
            np_name = os.path.join(dir, data_filenames[i].replace('.hdf5', ''))
            np.save(np_name, video)
            print('saved in', np_name + '.npy')



    def h5pt2npy(self, generate_npy=False, dir_name='train_label'):
        root = self.root
        train_folder = os.path.join(root, dir_name)

        if generate_npy == True:
            train_data_filenames = os.listdir(train_folder)
            train_data_filenames.sort(reverse=True)

            # train_data_filenames = train_data_filenames[520:]
            self.collect_save_data(train_folder, train_data_filenames)



    def get_labels(self, train_np_folder):
        train_data_filenames = os.listdir(train_np_folder)
        train_data_filenames.sort()

        train_label = list()
        for t in train_data_filenames:
            train_label.append(int(t.split('_')[1].replace("10", "0")))
            
        return train_label


    def check_npy_files(self, root):
        train_folder = os.path.join(root, 'train')

        train_np_folder = os.path.join(root, 'train_npy')
        if len(os.listdir(train_folder)) != len(os.listdir(train_np_folder)): 
            return False
        else:
            return True

    # def dataconvert(self, event_number,frame):
    #     self.train_dataset = np.full((80,128,128,event_number),0)
    #     for i in range(0,event_number):
    #         sample = self.train_np_data[i]
    #         self.train_dataset[:,:,:,i] =  sample[0:frame,:,:]  

    def dataconvert(self, event_number, frame, data):
        self.train_dataset = np.full((80, 128, 128, event_number),0)
        for i in range(0,event_number):
            sample = data[i]
            if np.shape(sample)[0] >= frame:
                self.train_dataset[:,:,:,i] =  sample[0:frame,:,:] 
            else:
                self.train_dataset[0:np.shape(sample)[0],:,:,i] =  sample   
        return self.train_dataset

    # def batch_generations(self,train_np_folder,times):
    #     if self.batch_true == False:
    #         for i in range(0,times):
    #             batch_folder = os.path.join(self.root, 'batch')
    #             batch_np_folder = os.path.join(batch_folder, str(i))
    #             if not os.path.exists(batch_np_folder):
    #                 os.mkdir(batch_np_folder) 

    #             batch_data_filenames = os.listdir(train_np_folder)
    #             batch_data_filenames.sort()   
    #             for j in range(1,11):
    #                 k = 0
    #                 while(j != int(batch_data_filenames[k].split('_')[1])):
    #                     k = k+1
    #                 path = os.path.join(train_np_folder, batch_data_filenames[k])
    #                 shutil.move(path, batch_np_folder)
    #     else:
    #         pass            


    def get_train_data(self, train_data_num, selected_event):
        self.event_num = len(selected_event)
        random.seed(0)
        train_data_folder = os.path.join(cfg.data_path, 'train_npy')
        train_filenames_all = os.listdir(train_data_folder)

        all_data_list = range(0, 98)
        selected_sample = random.sample(all_data_list, train_data_num)
        train_filenames = list()

        for filename in train_filenames_all:
            for event in selected_event:
                for sample in selected_sample:
                    match_str = "train_" + str(int(event)+1) + "_" + str(sample) + ".npy"
                    if match_str in filename:
                        train_filenames.append(filename)
        train_filenames.sort()

        cut_frame = 80

        train_data = list()
        train_label = list()

        # load np data and trancate 80 frame
        for filename in train_filenames:
            np_name = os.path.join(train_data_folder, filename)
            sample = np.load(np_name)
            event = int(filename.split("_")[-2])
            event_i = selected_event.index(event-1)
            
            if np.shape(sample)[0] >= cut_frame:
                train_data.append(sample[0:cut_frame, :, :] )
            else:
                data = np.zeros((cut_frame, 128, 128))
                data[0:np.shape(sample)[0], :, :] = sample
                train_data.append(data)
            train_label.append(event_i)
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        return train_data, train_label



    def get_test_data(self, test_data_num, selected_event):
        self.event_num = len(selected_event)
        random.seed(0)
        test_data_folder = os.path.join(cfg.data_path, 'test_npy')
        test_filenames_all = os.listdir(test_data_folder)

        all_data_list = range(0, 24) 
        selected_sample = random.sample(all_data_list, test_data_num)
        test_filenames = list()

        for filename in test_filenames_all:
            for event in selected_event:
                for sample in selected_sample:
                    match_str = "test_" + str(event+1) + "_" + str(sample) + ".npy"
                    if match_str in filename:
                        test_filenames.append(filename)
        
        test_filenames.sort()

        cut_frame = 80
        test_data = list()
        test_label = list()

        # load np data and trancate 80 frame
        for filename in test_filenames:
            np_name = os.path.join(test_data_folder, filename)
            sample = np.load(np_name)
            event = int(filename.split("_")[-2])
            event_i = selected_event.index(event-1)
            
            if np.shape(sample)[0] >= cut_frame:
                test_data.append(sample[0:cut_frame, :, :] )
            else:
                data = np.zeros((cut_frame, 128, 128))
                data[0:np.shape(sample)[0], :, :] = sample
                test_data.append(data)
            test_label.append(event_i)
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        return test_data, test_label



    def get_batch(self, train_data_num, batch_size, selected_event):
        self.event_num = len(selected_event)
        random.seed(0)
        train_data_folder = os.path.join(cfg.data_path, 'train_npy')
        train_filenames_all = os.listdir(train_data_folder)

        all_data_list = range(0, train_data_num)
        assert batch_size <= train_data_num
        selected_sample = random.sample(all_data_list, train_data_num)
        train_filenames = list()
        for filename in train_filenames_all:
            for event in selected_event:
                for sample in selected_sample:
                    match_str = "train_" + str(event) + "_" + str(sample) + ".npy"
                    if match_str in filename:
                        train_filenames.append(filename)

        selected_batch_sample = random.sample(selected_sample, batch_size)
        batch_filenames = list()
        cut_frame = 80
        
        batch_data = np.full((cut_frame, 128, 128, self.event_num), 0)
        for filename in train_filenames:
            for event_i, event in enumerate(selected_event):
                match_str = "train_" + str(event) + "_" + str(selected_batch_sample[0]) + ".npy"
                if match_str in filename:
                    # print(match_str, filename)
                    batch_filenames.append(filename)      
        batch_filenames.sort()
        
        # load np data and trancate 80 frame
        for filename in batch_filenames:
            np_name = os.path.join(train_data_folder, filename)
            sample = np.load(np_name)
            event = int(filename.split("_")[-2])
            event_i = selected_event.index(event)
            if np.shape(sample)[0] >= cut_frame:
                batch_data[:, :, :, event_i] =  sample[0:cut_frame, :, :] 
            else:
                batch_data[0:np.shape(sample)[0], :, :, event_i] =  sample   

        return batch_data, selected_event
             
if __name__ == "__main__":
    dataset = DatasetGesture(cfg.data_path)
    dataset.h5pt2npy(generate_npy=True)
    dataset.h5pt2npy(generate_npy=True, dir_name='test_label')

                   