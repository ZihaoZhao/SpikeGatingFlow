from train import SGF_train
from inference import SGF_inference
import os
import shutil
from expert import SGF_expert
import numpy as np
import cfg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib
from sklearn.decomposition import PCA
import argparse
import time

### This is a demonstration version of the Spike Gating Flow  ###

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='SGF 2.0.')
    parser.add_argument('--train', action='store_true', help='train SGF', default=False)
    parser.add_argument('--test', action='store_true', help='test SGF', default=False)
    parser.add_argument('--train_test', action='store_true', help='train and test SGF', default=False)
    parser.add_argument('--exp', type=str, help='event types', default='example')
    parser.add_argument('--train_data_num', type=int, help='train_data_num', default=98)
    parser.add_argument('--iter', type=int, help='iter', default=36)
    parser.add_argument('--save_excel', action='store_true', help='save_excel', default=True)
    parser.add_argument('--save_excel_path', type=str, help='save excel path', default='exp.xls')
    parser.add_argument('--save_train_curve', action='store_true', help='train knowledge curve', default=True)
    parser.add_argument('--each_sample_train_once', action='store_true', help='each_sample_train_once', default=False)
    parser.add_argument('--inner_batch_random', action='store_true', help='each_sample_train_once', default=False)
    parser.add_argument('--test_batch_list', type=str, help='test_batch_list', default='0')
    parser.add_argument('--if_print', action='store_true', help='if_print', default=False)

    parser.add_argument('--selected_events', type=str, help='event types', default='1+2+8+9+10_3_4+5_6+7')

    parser.add_argument('--st_paras', type=str, help='c_space_num, s_thre, t_window, t_thre', default='3_2_3_2')
    parser.add_argument('--save_st_core', action='store_true', help='save_excel', default=False)

    ## space expert parameters ##
    parser.add_argument('--resolution_s1', type=str, help='col, row', default='2_1')
    parser.add_argument('--thres_s1', type=str, help='space expert 1', default='20_10')
    parser.add_argument('--thres_bit_s1', type=int, help='col, row', default=4)
    parser.add_argument('--thres_step_s1', type=float, help='col, row', default=0.025)

    parser.add_argument('--resolution_s2', type=str, help='col, row', default='2_2')
    parser.add_argument('--thres_s2', type=str, help='space expert 2', default='3_120')
    parser.add_argument('--thres_bit_s2', type=int, help='col, row', default=3)
    parser.add_argument('--thres_step_s2', type=float, help='col, row', default=0.2)

    parser.add_argument('--resolution_s3', type=str, help='col, row', default='2_1')
    parser.add_argument('--thres_s3', type=str, help='space expert 3', default='17_8')
    parser.add_argument('--thres_bit_s3', type=int, help='col, row', default=1)
    parser.add_argument('--thres_step_s3', type=float, help='col, row', default=0.1)

    parser.add_argument('--resolution_s4', type=str, help='col, row', default='2_1')
    parser.add_argument('--thres_s4', type=str, help='space expert 2', default='18_10')
    parser.add_argument('--thres_bit_s4', type=int, help='col, row', default=1)
    parser.add_argument('--thres_step_s4', type=float, help='col, row', default=0.1)

    parser.add_argument('--resolution_s5', type=str, help='col, row', default='2_1')
    parser.add_argument('--thres_s5', type=str, help='space expert 2', default='22_8')
    parser.add_argument('--thres_bit_s5', type=int, help='col, row', default=1)
    parser.add_argument('--thres_step_s5', type=float, help='col, row', default=0.1)


    parser.add_argument('--vote_thres_step', action='store_true', help='vote_thres_step', default=False)

    parser.add_argument('--only_s1s2', action='store_true', help='only_s1s2', default=False)
    parser.add_argument('--no_s4', action='store_true', help='no_s4', default=False)
    
    parser.add_argument('--t_expert_skip', type=int, help='t_expert_skip', default=5)
    parser.add_argument('--t_expert_scale', type=int, help='t_expert_scale', default=50)
    
    parser.add_argument('--hopfield', action='store_true', help='hopfield', default=True)
    parser.add_argument('--hopfield_frame_para', type=str, help='40:3, 40:4, 50:5', default="40_3")
    parser.add_argument('--hf_skip', type=int, help='hf_skip', default=16)
    parser.add_argument('--hf_v_thres', type=float, help='hf_v_thres', default=0.5)
    parser.add_argument('--hf_h_thres', type=float, help='hf_h_thres', default=0.5)
    

    parser.add_argument('--reverse_inference', action='store_true', help='reverse_inference', default=False)
    parser.add_argument('--logic_inference', action='store_true', help='logic_inference', default=False)
    parser.add_argument('--pre_defined_logic', action='store_true', help='logic_inference', default=False)
    parser.add_argument('--use_unique_code', action='store_true', help='use_unique_code', default=False)
    parser.add_argument('--hist_predict', action='store_true', help='hist_predict', default=False)
    parser.add_argument('--know_weight_predict', action='store_true', help='hist_predict', default=False)
    parser.add_argument('--detailed_predict', action='store_true', help='detailed_predict', default=False)

    parser.add_argument('--code_mode', type=int, help='0:s1+s2+final; 1:s1+s2+act+final', default=3)
    parser.add_argument('--strict_knowledge', action='store_true', help='strict_knowledge', default=False)
    parser.add_argument('--all_knowledge', action='store_true', help='all_knowledge', default=False)
    parser.add_argument('--knowledge_distillation', action='store_true', help='knowledge_distillation', default=False)
    parser.add_argument('--sigmoid', action='store_true', help='sigmoid', default=False)

    parser.add_argument('--frame_skip_test', type=int, help='frame_skip', default=4)
    parser.add_argument('--threshold_test', type=float, help='hf_v_thres', default=0.7)
    parser.add_argument('--hist_threshold_test', type=float, help='hf_v_thres', default=5)
    parser.add_argument('--offset1_test', type=str, help='hf_v_thres', default="0_42")
    parser.add_argument('--offset2_test', type=str, help='hf_v_thres', default="0_42")

    parser.add_argument('--hop2_frame_skip', type=int, help='hop2_frame_skip', default=3)
    parser.add_argument('--hop2_threshold', type=float, help='hop2_threshold', default=0.5)
    parser.add_argument('--hop2_hist_threshold', type=int, help='hop2_hist_threshold', default=14)

    parser.add_argument('--test1', type=int, help='test1', default=60)
    parser.add_argument('--test2', type=int, help='test1', default=105)
    parser.add_argument('--test3', type=int, help='test1', default=35)
    parser.add_argument('--test4', type=int, help='test1', default=105)
    parser.add_argument('--test5', type=int, help='test1', default=6)
    parser.add_argument('--test6', type=int, help='test1', default=13)
    parser.add_argument('--test7', type=int, help='test1', default=7)

    # parser.add_argument()

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    localtime = time.asctime( time.localtime(time.time()) )
    parser.add_argument('--exp_date', type=str, help='localtime', default=str(localtime))
    print(str(localtime))


    args = parser.parse_args()
    print(str(args))
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        print("training...",args.exp)
        s1 = SGF_train(args)
        s1.sgf_train(data_num=args.train_data_num, iter=args.iter)

    elif args.test:
        print("testing...",args.exp)
        i1 = SGF_inference(args) 
        i1.sgf_inference()

    elif args.train_test:
        print("training...",args.exp)
        s1 = SGF_train(args)
        s1.sgf_train(data_num=args.train_data_num, iter=args.iter)
        print("testing...",args.exp)
        i1 = SGF_inference(args) 
        i1.sgf_inference()

    else:
        print("Choose --train or --test")
