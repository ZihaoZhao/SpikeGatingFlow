#! /bin/bash
###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : IBICAS, Fudan University
 # @Date         : 2021-08-07 20:36:58
 # @LastEditors  : Yanhong Wang
 # @LastEditTime : 2021-11-07 12:22:06
 # @FilePath     : /SGF_v2/dl_src/dl_scripts/c3d_train_num_36.sh
 # @Description  : 
### 
python="/yhwang/anaconda3/envs/sgf/bin/python"
dl_dir="/yhwang/0-Projects/1-snn"
dl_main="$dl_dir/main_dl.py"
result_dir="/yhwang/0-Projects/1-snn/dl_src/dl_results"

${python} ${dl_main} \
--epochs 300 \
--net "c3d" \
--train_data_num 36