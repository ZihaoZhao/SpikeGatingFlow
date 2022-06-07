import cfg
import argparse
import time
import os

from dvsgesture_t import DatasetGesture
from dvsgesture_i import DatasetGesture_i

import sys
sys.path.append("dl_src")
from dl_src.cnn import ConvClassifier

### Deep Learning Counter Part  ###


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='SGF DL 1.0.')
    parser.add_argument('--train_data_num', type=int, help='train_data_num,1,2,3,6,12,24,36,48,60,72,84,90,96', default=1)
    parser.add_argument('--test_data_num', type=int, help='test_data_num', default=24)
    parser.add_argument('--epochs', type=int, help='epochs', default=1)
    parser.add_argument('--selected_events', type=list, help='event types', default=[0,1,2,3,4,5,6,7,8,9])

    # # Test setting
    # parser = argparse.ArgumentParser(description='SGF DL 1.0.')
    # parser.add_argument('--train_data_num', type=int, help='train_data_num', default=10)
    # parser.add_argument('--test_data_num', type=int, help='test_data_num', default=1)
    # parser.add_argument('--epochs', type=int, help='epochs', default=1)
    # parser.add_argument('--selected_events', type=list, help='event types', default=[0,1,2,3,4])


    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    parser.add_argument('--lr', type=float, help='lr', default=0.01)
    parser.add_argument('--log_dir', type=str, help='log directory', \
                                default=None)
    parser.add_argument('--net', type=str, help='"convnet", "c3d", "i3d"', default="convnet")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)

    train_cfg = parser.parse_args()
    print(str(train_cfg))
    return train_cfg


if __name__ == "__main__":
    train_cfg = parse_args()
    tm = time.localtime()
    tm_data = f"{tm.tm_year}{'%02d'%(tm.tm_mon)}{'%02d'%(tm.tm_mday)}"
    tm_time = f"{'%02d'%(tm.tm_hour)}{'%02d'%(tm.tm_min)}{'%02d'%(tm.tm_sec)}"
    if train_cfg.log_dir == None:
        train_cfg.log_dir = f"/yhwang/0-Projects/1-snn/dl_src/dl_results/{train_cfg.net}_train_num_{train_cfg.train_data_num}/{tm_data}_{tm_time}"
    print(train_cfg.log_dir)
    if not os.path.exists(train_cfg.log_dir):
        os.makedirs(train_cfg.log_dir)
    selected_event = train_cfg.selected_events
    dataset_t = DatasetGesture(cfg.data_path)            ## Read training data
    dataset_i = DatasetGesture_i(cfg.data_path)    ## Read the inference dataset

    
    from dl_src.model_cfg import cnn_cfg
    cnn_cfg.class_num = len(selected_event)
    print("CNN Configration:", cnn_cfg)
    print("============Initing CNN model")
    cnn_model = ConvClassifier(cnn_cfg, train_cfg)
    cnn_model.to(train_cfg.device)

    print("============Loading data=============")
    train_data, train_label = dataset_t.get_train_data(train_data_num=train_cfg.train_data_num, \
                                                    selected_event=selected_event)
    # from dl_src.utils import visualize_batch_data
    # visualize_batch_data(train_data[:, ::5, :, :], train_label, save_dir="/yhwang/0-Projects/1-snn/dl_src/dl_visualize2")

    
    train_data, train_label = cnn_model.get_batch_data(train_data, train_label, 1)
    if train_cfg.net == "c3d":
        train_data = cnn_model.resize_data(train_data, cnn_cfg.frame_scale, cnn_cfg.feature_scale)
    else:
        train_data = cnn_model.resize_data(train_data, cnn_cfg.frame_scale)
    
    # from dl_src.utils import visualize_batch_data
    # for i, (batch_data, batch_label) in enumerate(zip(train_data, train_label)):
    #     visualize_batch_data(batch_data[:, ::3, :, :], batch_label, \
    #         save_dir="/yhwang/0-Projects/1-snn/dl_src/dl_visualize1", git_end_str=f"_batch{i}")


    test_data, test_label = dataset_t.get_test_data(test_data_num=train_cfg.test_data_num, \
                                                selected_event=selected_event)
    test_data, test_label = cnn_model.get_batch_data(test_data, test_label, train_cfg.test_data_num)             
    if train_cfg.net == "c3d":
        test_data = cnn_model.resize_data(test_data, cnn_cfg.frame_scale, cnn_cfg.feature_scale)
    else:
        test_data = cnn_model.resize_data(test_data, cnn_cfg.frame_scale)



    # print("============Training")
    # cnn_model.train(train_data, train_label, epochs=args.epochs)

    # print("============Testing")
    # test_acc, test_loss = cnn_model.test(test_data, test_label)
    # print("test_acc", test_acc, "test_loss", test_loss)

    print("============Training and Testing=============")
    cnn_model.train_test(train_data, train_label, test_data, test_label, train_cfg.epochs, train_cfg.log_dir)
