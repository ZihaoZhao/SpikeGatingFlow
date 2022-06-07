import torch
import numpy as np

from visualization_utils import save_visualize_gif


def visualize_batch_data(x, y, save_dir, git_end_str=""):
    # x:[b_size, frame, h, w]
    for b_cnt in range(x.shape[0]):
        input_list = list()
        input_tensor = x[b_cnt]
        for frame_cnt in range(input_tensor.shape[0]):
            input_list.append(input_tensor[frame_cnt])
        input_shape = tuple(input_list[0].shape)
        if type(y[b_cnt])==torch.Tensor:
            label = int(torch.where(y[b_cnt]==1)[0])
        elif type(y[b_cnt])==np.int64:
            label = int(y[b_cnt])
        # image_name = f"/yhwang/0-Projects/1-snn/dl_src/dl_visualize/label{label}_train_batch{b_i}_data{b_cnt}.gif"
        image_name = f"{save_dir}/label{label}_data{b_cnt}{git_end_str}.gif"
        save_visualize_gif(input_list, input_shape, image_name)