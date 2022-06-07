#! /bin/bash
###
 # @Author       : Zihao Zhao
 # @E-mail       : zhzhao18@fudan.edu.cn
 # @Company      : IBICAS, Fudan University
 # @Date         : 2021-08-07 20:36:58
 # @LastEditors  : Zihao Zhao
 # @LastEditTime : 2021-09-05 22:30:58
 # @FilePath     : /SGF_v2/script/10_2122.sh
 # @Description  : 
### 
python = "/yhwang/anaconda3/envs/sgf/bin/python"
dl_dir = "/yhwang/0-Projects/1-snn/dl_src"
dl_main = "$dl_dir/main_dl.py"
result_dir = "/yhwang/0-Projects/1-snn/dl_src/dl_results"

cd $dl_main

event_list=("1_2_3_4_5_6_7_8_9_10")

for event in ${event_list[*]}
do 
    ${python} ${main} \
    --selected_events ${event} \
    --epochs 1
done