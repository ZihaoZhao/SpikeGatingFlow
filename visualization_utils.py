import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import cv2
import imageio
import os
from sklearn.decomposition import PCA

import cfg

def visualize(input, input_shape):
    input_image = input.reshape(input_shape)
    plt.matshow(input_image, cmap='hot')
    plt.colorbar()
    plt.show()

def pca_decomposition(input_data, target_dim):
    pca=PCA(n_components=target_dim)
    pca.fit(input_data)
    pca_data = pca.transform(input_data)
    plt.scatter(pca_data[:,0],pca_data[:,1])
    plt.savefig(cfg.code_path + "/output/pca_test.jpg", dpi=300)
    plt.clf()
    return pca_data

def save_visualize(input, input_shape, image_name):
    if type(input_shape) == type((1, 1)):
        input_image = input.reshape(input_shape)
        plt.matshow(input_image, cmap='hot')
        # plt.matshow(input_image, cmap='hot', vmin = 0, vmax = 1)
        plt.colorbar()
        plt.savefig(image_name, dpi=300)
    elif type(input_shape) == type(1):
        input_image = input.reshape((input_shape, 1))
        plt.matshow(input_image, cmap='hot')
        # plt.colorbar()
        plt.savefig(image_name)
    else:
        assert 0, "save error."

def save_curve(x, y, image_name):
    plt.plot(x,y)
    plt.savefig(image_name)
    plt.clf()
       
def save_vis_formatted(train_data):
    for event in range(train_data.shape[-1]):
        save_dir = cfg.code_path + "/output/"+str(event)
        try:
            os.mkdir(save_dir)
            print(save_dir)
        except FileExistsError:
            print(save_dir)
        for frame in range(train_data.shape[0]):
            save_visualize(train_data[frame, :, :, event], (128,128), 
                            os.path.join(save_dir, str(frame)+".jpg")) 

def train_result_vis_pca(id_sel):

    info = np.load(cfg.code_path + "/expert4_information.npy")
    pca=PCA(n_components=2)
    pca.fit(info)
    pca_data = pca.transform(info)
    id = np.load(cfg.code_path + "/expert4_id.npy")

    plt.xlim(xmax=2,xmin=-2)
    plt.ylim(ymax=2,ymin=-2)
    if id_sel > 0:
        pca_data_1 = list()
        for i in range(0, info.shape[0]):
            if id[i] == id_sel:
                print(i, id[i], pca_data[i], info[i])
                pca_data_1.append(pca_data[i])
        pca_data_1 = np.array(pca_data_1)

        plt.scatter(pca_data_1[:,0], pca_data_1[:,1])
        plt.savefig(cfg.code_path + "/output/pca_test_"+str(id_sel)+".jpg",dpi=300)
        plt.clf()
    else:
        plt.scatter(pca_data[:,0], pca_data[:,1], c=id*30)
        plt.savefig(cfg.code_path + "/output/pca_test_all.jpg",dpi=300)
        plt.clf()

def train_result_vis_pca_3d(id_sel):
    info = np.load(cfg.code_path + "/expert4_information.npy")
    pca=PCA(n_components=3)
    pca.fit(info)
    pca_data = pca.transform(info)
    id = np.load(cfg.code_path + "/expert4_id.npy")

    # plt.xlim(xmax=2,xmin=-2)
    # plt.ylim(ymax=2,ymin=-2)
    # plt.zlim(zmax=2,zmin=-2)
    pca_data_1 = list()
    for i in range(0, info.shape[0]):
        if id[i] == id_sel:
            print(i, id[i], pca_data[i], info[i])
            pca_data_1.append(pca_data[i])
    pca_data_1 = np.array(pca_data_1)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,2], c=id*30)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.savefig(cfg.code_path + "/output/pca_3d_test_"+str(id_sel)+".jpg",dpi=300)
    fig.clf()

def save_visualize_gif(input_list, input_shape, image_name):
    # fig = plt.figure()
    # camera = Camera(fig)
    # print("Saving gif", image_name)
    # for input in input_list:
    #     input_image = input.reshape(input_shape)
    #     plt.matshow(input_image, cmap='hot')
    #     plt.colorbar()
    #     plt.show()
    #     # plt.savefig(image_name)
    #     camera.snap()
    # animation = camera.animate()
    # animation.save(image_name)
    # # plt.show()    
    frames = []
    print("Saving", image_name)
    if type(input_shape) == type((1, 1)):
        for input in input_list:
            input_image = input.reshape(input_shape)
            plt.matshow(input_image, cmap='hot')
            # plt.colorbar()
            plt.savefig("tmp.png")
            frames.append(cv2.imread("tmp.png"))
        imageio.mimsave(image_name, frames, fps=20)
    elif type(input_shape) == type(1):
        for input in input_list:
            input_image = input.reshape((input_shape, 1))
            plt.matshow(input_image, cmap='hot')
            # plt.colorbar()
            plt.savefig("tmp.png")
            frames.append(cv2.imread("tmp.png"))
        imageio.mimsave(image_name, frames, fps=256)
    # gif.save(frames, 'random.gif', duration=50)

def save_visualize_img_gif(input_list, image_name):
    frames = []
    print("Saving", image_name)
    for input in input_list:
        frames.append(input)
    imageio.mimsave(image_name, frames, fps=25)

def save_visualize_3d(input_list, input_shape, image_name):
    x = []
    y = []
    z = []
    # c = []
    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    ax.view_init(elev=10., azim=11)

    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, 1024))

    for i, input in enumerate(input_list):
        input_image = input.reshape(input_shape)
        input_max = input.max()
        for j in range(input_image.shape[0]):
            for k in range(input_image.shape[1]):
                if input_image[j][k] != 0:
                    x.append(j)
                    z.append(k)
                    y.append(i)
                    # c.append(colors[int(input_image[j][k]/ input_max *255)])
                    # 
                    # print(colors[int(input_image[j][k]/ input_max *255)])
    # ax.scatter(x,y,z,c,'filled',cmap='spectral')
    ax.scatter(x,y,z,c='b',marker='.',s=20,linewidth=0,alpha=0.8,cmap='spectral')
        # plt.matshow(input_image, cmap='hot')
        # plt.matshow(input_image, cmap='hot', vmin = 0, vmax = 1)
        # plt.colorbar()
    plt.savefig(image_name, dpi=300)

def save_visualize_3dsurface(input, input_shape, image_name):
    figure = plt.figure()
    ax = Axes3D(figure,azim=-75,elev=30)
    X = np.arange(0,input_shape[1],1)
    Y = np.arange(0,input_shape[0],1)
    X,Y = np.meshgrid(X,Y)
    ax.plot_surface(X,Y,input,rstride=1,cstride=1,cmap='rainbow')
    
    plt.savefig(image_name, dpi=300)

def show_wave(wave):
    dist = 60
    channel_num = 20
    y_range = dist * channel_num + 50
    start_time = int(0 * 30000)
    time_scale = 15
    i=0
    plt.cla()
    plt.xlim(i, i + 600)
    plt.ylim(-y_range / 2 / dist + channel_num / 2, y_range / 2 / dist + channel_num / 2)
    x = np.linspace(i, i + 600, 601).astype(int)
    for j in range(0, channel_num):
        y = wave[j, x * time_scale + start_time] * 0.4 * 1e6 + int(j - channel_num / 2) * dist + dist / 2
        y = y / dist + channel_num / 2
        plt.plot(x, y)
    i = i + 1

    plt.xlabel("Sample Data Point", size=14)
    plt.ylabel("Recording Channel", size=14)

    plt.title("EEG Motor Movement/Imagery Dataset",
                fontdict={'family': 'serif',
                        'color': 'darkgreen',
                        'weight': 'bold',
                        'size': 18})

    plt.savefig("tmp.png")

def save_wave(wave, image_name, title):
    dist = 60
    channel_num = 20
    y_range = dist * channel_num + 50
    start_time = int(0 * 30000)
    time_scale = int(wave.shape[1]/600)
    i=0
    plt.cla()
    plt.xlim(i, i + 600)
    plt.ylim(-y_range / 2 / dist + channel_num / 2, y_range / 2 / dist + channel_num / 2)
    x = np.linspace(i, i + 600, 601).astype(int)
    for j in range(0, channel_num):
        y = wave[j, x * time_scale + start_time] * 0.4 * 1e6 + int(j - channel_num / 2) * dist + dist / 2
        y = y / dist + channel_num / 2
        plt.plot(x, y)
    i = i + 1

    plt.xlabel("Sample Data Point", size=14)
    plt.ylabel("Recording Channel", size=14)

    plt.title(title,
                fontdict={'family': 'serif',
                        'color': 'darkgreen',
                        'weight': 'bold',
                        'size': 18})

    plt.savefig(image_name)