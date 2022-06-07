import argparse

parser = argparse.ArgumentParser(description='CNN configuration.')
parser.add_argument('--in_channels', type=list, help='in_channels', default=[])
parser.add_argument('--out_channels', type=list, help='out_channels', default=[])
parser.add_argument('--kernels', type=list, help='kernels', default=[])
parser.add_argument('--strides', type=list, help='strides', default=[])
parser.add_argument('--pads', type=list, help='pads', default=[])
parser.add_argument('--groups', type=list, help='groups', default=[])
parser.add_argument('--class_num', type=int, help='class_num', default=10)

cnn_cfg = parser.parse_args()

cnn_cfg.in_channels = [80, 12, 252, 256, 256, 512, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 968]
cnn_cfg.out_channels = [12, 252, 256, 256, 512, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 968, 2640]
cnn_cfg.kernels = [3, 4, 1, 2, 3, 1, 1, 1, 2, 3, 1, 1, 2, 1, 1, 1]
cnn_cfg.strides = [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
cnn_cfg.pads = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
cnn_cfg.groups = [1, 2, 2, 2, 32, 4, 4, 4, 16, 64, 8, 8, 32, 8, 8, 8]