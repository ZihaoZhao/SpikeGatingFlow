import numpy as np
import cfg

infer_list = list()
ref_result = [10]*24 + [1]*24 +[8]*24 + [9]*24
for fs in [1,2,3,4,5]:
    for thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
        for hist_thres in range(2,17):
# for fs in [1,2,3,4,5]:
#     for thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]:
#         for hist_thres in range(4,15):
            exp_name = "8_1910_hop_train36_{:}_{:}_{:}".format(fs, thres, hist_thres)
            # if exp_name not in ["8_1910_hop_train36_2_0.6_7",
            #                     "8_1910_hop_train36_2_0.2_7",
            #                     "8_1910_hop_train36_4_0.1_9",
            #                     "8_1910_hop_train36_4_0.3_3",
            #                     "8_1910_hop_train36_2_0.4_7",
            #                     "8_1910_hop_train36_3_0.6_6",
            #                     "8_1910_hop_train36_5_0.3_6",
            #                     "8_1910_hop_train36_3_0.5_11",
            #                     "8_1910_hop_train36_3_0.6_7"]:
            #     continue
            try:
                infer_result = np.loadtxt("/Users/zzh/Code/SGF_v2/data/" + exp_name + "/predict_list.txt")
                # print(cfg.code_path + "/data/" + exp_name + "/predict_list.txt")
                infer_list.append(infer_result)
            except:
                print("miss ", exp_name)
infer_result_array = np.array(infer_list)
print(infer_result_array.shape)

correct_cnt = 0
for i in range(0,97):
    cnt_1 = 0
    cnt_8 = 0
    cnt_9 = 0
    cnt_10 = 0
    for pre_i, pre in enumerate(infer_result_array[:,i]):
        if pre == 1:
            cnt_1 += 1
        elif pre == 8:
            cnt_8 += 1
        elif pre == 9:
            cnt_9 += 1
        elif pre == 10:
            cnt_10 += 1
    cnt_max = max(cnt_1, cnt_8, cnt_9, cnt_10)
    if cnt_1 == cnt_max and ref_result[i] == 1:
        correct_cnt += 1
    elif cnt_8 == cnt_max and ref_result[i] == 8:
        correct_cnt += 1
    elif cnt_9 == cnt_max and ref_result[i] == 9:
        correct_cnt += 1
    elif cnt_10 == cnt_max and ref_result[i] == 10:
        correct_cnt += 1
print(correct_cnt, correct_cnt/96)