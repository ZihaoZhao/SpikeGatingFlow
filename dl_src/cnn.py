# reference:https://github.com/shanglianlm0525/PyTorch-Networks/blob/f8b0376ba6a0dcfd5d461fcbd1cbd7c236a944b4/SemanticSegmentation/SegNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
import random

from collections import OrderedDict
from visualization_utils import save_visualize_gif


def Conv2DBNReLU(in_channels, out_channels, kernel_size, stride, pad, groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                kernel_size=kernel_size, stride=stride, padding=pad, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )


def Conv2D(in_channels, out_channels, kernel_size, stride, pad, groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, groups=groups),
        )


# class ConvClassifier(nn.Module):
#     def __init__(self, in_channels, out_channels, kernels, strides, pads, groups, class_num, device):
#         super().__init__()
#         convnet = OrderedDict()
#         for l in range(len(in_channels)):
#             convnet[f"conv_{l}"] = Conv2DBNReLU(in_channels[l], out_channels[l], kernels[l], strides[l], pads[l], groups[l])
#         convnet["flatten"] = nn.Flatten()
#         convnet["fc"] = nn.Linear(39600*15, class_num)
#         convnet["sigmoid"] = nn.Sigmoid()
#         self.ConvNet = nn.Sequential(convnet)
#         self.ConvNet = self.ConvNet.float()
#         self.class_num = class_num
#         self.sgd_optimizer = optim.SGD(self.ConvNet.parameters(), lr = 0.00001, momentum = 0.9)
#         self.criterion = nn.MSELoss()
#         self.device = device


#     def forward(self, x):
#         # print(self.ConvNet)
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
#                 kernel_size=kernel_size, stride=stride, padding=pad, groups=groups),
#         )


class I3DNet64(nn.Module):
    def __init__(self, args):
        super().__init__()


class C3DNet128(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 3Dkernel: IC,OC,k1,k2,k3
        # 4D Input: B,IC,F,H,W
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))


        self.conv6a = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv6b = nn.Conv3d(1024, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))

        # self.fc6 = nn.Linear(8192, 4096)
        self.fc6 = nn.Linear(1024*3*3*3, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, args.class_num)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()



    def forward(self, x):
        if x.dim()==4:
            x = x.unsqueeze(1)
        # 这里输入为[1,1,80,128,128]
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        # 这里输出为[1,512,5,5,5]
        # add by wyh
        # h = self.relu(self.conv6a(h))
        # h = self.relu(self.conv6b(h))
        # h = self.pool6(h)
        # 这里输出为[1,1024,2,3,3]

        h = h.view(-1, 1024*3*3*3)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs



class C3DNet64(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 3Dkernel: IC,OC,k1,k2,k3
        # 4D Input: B,IC,F,H,W
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(512*1*3*3, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, args.class_num)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()



    def forward(self, x):
        if x.dim()==4:
            x = x.unsqueeze(1)
        # 这里输入为[1,1,80,128,128]
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 512*1*3*3)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs

class ConvNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernels = args.kernels
        strides = args.strides
        pads = args.pads
        groups = args.groups
        class_num = args.class_num
        convnet = OrderedDict()
        for l in range(len(in_channels)-4):
            convnet[f"conv_{l}"] = Conv2DBNReLU(in_channels[l], out_channels[l], kernels[l], strides[l], pads[l], groups[l])
        convnet["pooling"] = nn.MaxPool2d(kernel_size=3, stride=3)
        convnet["flatten"] = nn.Flatten()
        convnet["conv14"] = nn.Linear(1024, 1024)
        convnet["conv14"] = nn.Linear(1024, 1024)
        convnet["conv15"] = nn.Linear(1024, 968)
        convnet["conv16"] = nn.Linear(968, 2640)
        convnet["fc"] = nn.Linear(2640, class_num)
        convnet["act"] = nn.Softmax()
        self.convnet = nn.Sequential(convnet)
        
    def forward(self, x):
        return self.convnet(x)


class ConvClassifier(nn.Module):
    def __init__(self, cnn_cfg, train_cfg):
        super().__init__()
        self.classifierNet = self.create_model(cnn_cfg, train_cfg)
        self.classifierNet = self.classifierNet.float()
        self.class_num = cnn_cfg.class_num
        self.sgd_optimizer = optim.SGD(self.classifierNet.parameters(), \
                                        lr = train_cfg.lr, \
                                        momentum = train_cfg.momentum)
        self.criterion = nn.MSELoss()
        self.device = train_cfg.device


    def create_model(self, cnn_cfg, train_cfg):
        if train_cfg.net == "convnet":
            return ConvNet(cnn_cfg)
        elif train_cfg.net == "c3d":
            return C3DNet64(cnn_cfg)
        elif train_cfg.net == "i3d":
            return I3DNet64(cnn_cfg)
        else:
            NotImplementedError



    def forward(self, x):
        # print(self.classifierNet)
        # cuda_used = torch.cuda.memory_allocated()
        # cuda_reserved = torch.cuda.memory_reserved()
        # print(f"\tCuda used {cuda_used}/{cuda_reserved}")
        x = x.to(self.device)
        y = self.classifierNet(x)
        # for layer in range(len(self.classifierNet)):
        #     x = self.classifierNet[layer](x)
        #     print(layer, x.shape)
        return y


    def cal_loss(self, y_pred, y_b):
        self.sgd_optimizer.zero_grad()
        y_b = y_b.to(self.device)
        loss = self.criterion(y_pred, y_b)
        return loss


    def train(self, x, y):
        # x:b,ic,h,w   y:b,1,class_num
        shuffle_batch_idx = random.sample(range(len(x)), len(x))
        for b_i, ori_i in enumerate(shuffle_batch_idx):
            x_b = x[ori_i]
            y_b = y[ori_i]
            y_pred = self.forward(x_b)
            if (b_i+1)%100==0:
                print(f"\tTraining iteration:{b_i+1}/{len(x)}")
            # print(f"x_b:{x_b.sum()}")
            # print(f"y_pred_shape:{y_pred.shape}, y_shape:{y_b.shape}")
            # print(f"y_pred:{y_pred}, \n y:{y_b}")
            loss = self.cal_loss(y_pred, y_b)
            loss.backward()
            # print(self.classifierNet[17].weight.grad.sum())
            self.sgd_optimizer.step()
            # for b_cnt in range(x_b.shape[0]):
            #     input_list = list()
            #     input_tensor = x_b[b_cnt].detach().cpu()
            #     for frame_cnt in range(input_tensor.shape[0]):
            #         input_list.append(input_tensor[frame_cnt])
            #     input_shape = tuple(input_list[0].shape)
            #     label = int(torch.where(y_b.detach().cpu()[b_cnt]==1)[0])
            #     image_name = f"/yhwang/0-Projects/1-snn/dl_src/dl_visualize/label{label}_train_batch{b_i}_data{b_cnt}.gif"
            #     save_visualize_gif(input_list, input_shape, image_name)

        train_acc, train_loss = self.test(x, y)
        return train_acc, train_loss

    def test(self, x, y):        
        y_pred_list = list()
        loss_list = list()
        acc_list = list()
        for b_i in range(len(x)):
            x_b = x[b_i]
            y_b = y[b_i]
            y_pred = self.forward(x_b)
            loss = self.cal_loss(y_pred, y_b)
            loss_list.append(loss.to("cpu").detach().numpy())
            y_label_class = y_b.argmax(dim=1)
            y_pred_class = y_pred.cpu().argmax(dim=1)
            acc_list.append(torch.sum(y_label_class==y_pred_class)/len(y_pred_class))
        test_acc = np.mean(acc_list)
        test_loss = np.mean(loss_list)
        return test_acc, test_loss


    def train_test(self, train_x, train_y, test_x, test_y, epochs, log_dir):
        print("log directory", log_dir)
        writer = SummaryWriter(log_dir)
        train_acc_list = list()
        train_loss_list = list()
        test_acc_list = list()
        test_loss_list = list()
        for e in range(epochs):
            print(f"-------------------------Epoch {e+1}/{epochs} ---------------------")
            train_acc, train_loss = self.train(train_x, train_y)
            test_acc, test_loss = self.test(test_x, test_y)
            writer.add_scalar("train/acc", train_acc, e)
            writer.add_scalar("train/loss", train_loss, e)
            writer.add_scalar("test/acc", test_acc, e)
            writer.add_scalar("test/loss", test_loss, e)
            print(f"Train: acc={train_acc}|loss={train_loss}")
            print(f"test: acc={test_acc}|loss={test_loss}")
            with open(f"{log_dir}/{self.class_num}_epoch{epochs}.txt", "a") as f:
                f.write(f"-------------------------Epoch {e+1}/{epochs} ---------------------\n")
                f.write(f"Train: acc={train_acc}|loss={train_loss}\n")
                f.write(f"test: acc={test_acc}|loss={test_loss}\n")
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        model_name = f"{log_dir}/{self.class_num}_epoch{epochs}.pkl"
        torch.save(self.classifierNet, model_name)


    def get_batch_data(self, x, y, batch_size=1):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).float()

        x_batch = list()
        y_batch = list()
        if x.dim()==4:
            shuffle_idx = torch.randperm(x.shape[0])
            x = x[shuffle_idx, :, :, :]
            assert x.shape[0]%batch_size == 0
            for b_idx in range(0, x.shape[0], batch_size):
                x_batch.append(x[b_idx:b_idx+batch_size,:,:,:])
        # size, batch_size, class_num  98,1,10
        if y.dim()==1:
            assert y.shape[0]%batch_size == 0
            for b_idx in range(int(x.shape[0]/batch_size)):
                y_batch.append(torch.zeros(batch_size, self.class_num))
                for idx in range(batch_size):
                    y_idx = int(shuffle_idx[b_idx*batch_size+idx])
                    y_batch[b_idx][idx][int(y[y_idx])] = 1
                    # print(y_batch[b_idx].shape)
        assert len(x_batch) == len(y_batch)
        return x_batch, y_batch

    def resize_data(self, x, frame_scale=1, feature_scale=1):
        x_new = list()
        frame_interval = int(1/frame_scale)
        for x_b in x:
            # print(x_b.shape)
            x_b_resize = F.interpolate(x_b, None, feature_scale, mode='bilinear')
            x_new.append(x_b_resize[:, ::frame_interval, :, :])
        return x_new
