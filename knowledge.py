from os import remove
import numpy as np
import cfg
import copy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
class Knowledge(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def knowledge_base(self, information, label, event_pre_id, expert1_pre_knowledge, ignore_list=[]):
        self.events = list()
        for e in self.args.selected_events.split("_"):
            self.events.extend(e.split("+"))
        # print("start decoding from active location areas:")
        target = information  # the outputs from space expert1
        s_results = self.check_similarity(target, label, self.args.selected_events.split("_"))
        np.fill_diagonal(s_results, 1)
        expert1_index= np.linspace(0,0,np.shape(s_results)[0])
        for i in range(0,np.shape(s_results)[0]):
            expert1_index[i] = 1
            for j in range(0,np.shape(s_results)[1]):
                if not label[j] in ignore_list and s_results[i][j] == 0:
                    expert1_index[i] = 0
            # if np.all(s_results[:,i] == 1):
            #     expert1_index[i] = 1
            # else:
            #     expert1_index[i] = 0        
        useful_id = np.where(expert1_index ==1)[0]  
        expert1_knowledge =  [0 for i in range (np.shape(target)[0])]
        event_post_id =  event_pre_id
        expert1_post_knowledge = expert1_pre_knowledge
        similarity = self.check_similarity(information, label, self.args.selected_events.split("_"))
        np.fill_diagonal(similarity, 1)  ## fill diagnoal to 1
        # print(similarity)
        if self.args.strict_knowledge:
            if np.all(similarity) == 1:
                for i in range(0,np.shape(useful_id)[0]):
                    event_id = label[useful_id[i]]                         # get event id
                    expert1_knowledge = target[:,useful_id[i]]
                    event_post_id = np.append(event_post_id, event_id)      
                    expert1_post_knowledge = np.vstack([expert1_post_knowledge, expert1_knowledge])
            else:
                k1, id1 = expert1_pre_knowledge, event_pre_id
        elif self.args.all_knowledge:
            for id_i, event_id in enumerate(label):
                event_post_id = np.append(event_post_id, event_id)      
                expert1_post_knowledge = np.vstack([expert1_post_knowledge, information[:,id_i]])
        else:
            for i in range(0,np.shape(useful_id)[0]):
                event_id = label[useful_id[i]]                         # get event id
                expert1_knowledge = target[:,useful_id[i]]
                event_post_id = np.append(event_post_id, event_id)      
                expert1_post_knowledge = np.vstack([expert1_post_knowledge, expert1_knowledge])

        if len(event_post_id) == 1 and event_post_id[0] == -1:
            k1, id1 = expert1_pre_knowledge, event_pre_id
        else:
            k1, id1 = self.knowledge_distillation(event_post_id, expert1_post_knowledge)  
            # k1, id1 = expert1_post_knowledge, event_post_id

            # for knowledge in k1:
            #     if np.all(knowledge == np.array([0,0,0,0,0,1])):
            #         print(knowledge)
        return k1, id1

    def check_similarity(self, data, label, select_event=None):
        if data.shape[0] == data.size:
            data = data.reshape(data.shape[0], 1)
        similarity = np.full((np.shape(data)[1],np.shape(data)[1]),0)

        macro_class_dict = dict()
        for l in label:
            for macro_class in select_event:
                if l in macro_class.split("+"):
                    macro_class_dict[l] = macro_class
                    break

        for i in range(0, np.shape(data)[1]):   ## event
           for j in range(0, np.shape(data)[1]): ## event
               if np.all(data[:,i] == data[:,j]) and macro_class_dict[label[i]] != macro_class_dict[label[j]]: 
                   similarity[i,j] = 0
               else:
                   similarity[i,j] = 1    
        return similarity  

    def knowledge_distillation(self, event_post_id, expert1_post_knowledge):

        if self.args.knowledge_distillation:
            s_results = self.check_similarity(np.transpose(expert1_post_knowledge), self.args.selected_events.split("_"))
            np.fill_diagonal(s_results, 1)
            il = np.tril_indices(np.shape(s_results)[0])
            s_results[il] = 1
            for i in range(0,np.shape(s_results)[0]):
                if np.any(s_results[i,:] == 0):
                    print("start knowledge distillation")
                    index = np.where(s_results[i,:] == 0)[0]
                    for j in range(0,np.shape(index)[0]):                
                        expert1_post_knowledge[index[j],:] = -1
                        event_post_id[index[j]] =  -1

            expert1_post_knowledge = expert1_post_knowledge[(expert1_post_knowledge >=0).all(axis = 1)] 
            event_post_id = [id for id in event_post_id if id not in ['-1',-1]]
        # event_post_id = event_post_id.reshape((len(event_post_id), 1))
        # event_post_id = event_post_id[event_post_id >=0]       
        else: 
            return expert1_post_knowledge, event_post_id
        return  expert1_post_knowledge, event_post_id

    # def knowledge_classifications(self,k):
    #     raw_data = np.load(k)   
    #     number = np.shape(raw_data)[0]
    #     pattern_length = np.shape(raw_data)[1]
    #     pattern =[]
    #     pattern_id = [-1 for i in range(0,number)]
    #     index = 0
    #     for i in range (0, number):
    #         flag = 0
    #         j = 0
    #         while flag == 0:
    #             if i == j or np.all(raw_data[i,:] == 0):
    #                 pass

    #             elif np.array_equal(raw_data[i,:], raw_data[j,:]) == True:
    #                 if pattern_id[j] != -1:
    #                     pattern_id[i] = pattern_id[j]
    #                     flag = 1
    #                 elif pattern_id[j] == -1 and pattern_id[i] == -1:
    #                     pattern_id[i] = index 
    #                     flag = 1
    #                     index = index +1 
    #                 else:
    #                     pass
                       
    #             else:
    #                 pass    
    #             j = j+1  
    #             if j == number and flag == 0:
    #                 if np.all(raw_data[i,:] == 0):
    #                     flag = 1
    #                 else:    
    #                     flag = 1 
    #                     pattern_id[i] = index  
    #                     index = index +1 

    #     ## record pattern Id and the weights.  
    #     pattern_num = np.amax(pattern_id)
    #     pattern_weight = [0 for i in range(0,pattern_num)]
    #     for i in range(0, pattern_num):
    #         for j in range(0, number):
    #             if i == pattern_id[j]:
    #                 pattern_weight[i] = pattern_weight[i] +1

    #     ## record pattern information
    #     pattern_information = np.full((pattern_num,pattern_length),0)
    #     for i in range(0,pattern_num):
    #         flag = 0
    #         j = 0
    #         while flag == 0:
    #             if i == pattern_id[j]:
    #                 pattern_information[i,:] = raw_data[j,:]
    #                 flag = 1
    #             else:
    #                 pass
    #             j = j +1        

    #     return pattern_weight,pattern_information




    # def knowledge_inference(self, test_data, k2,kw2):
    #     k1 = np.load(test_data)
    #     test_num = np.shape(k1)[0]

    #     know_leng = np.shape(k1)[1]

    #     test_result = [0 for i in range (0, test_num)]
    #     know_num = np.shape(k2)[0]
    #     kw2 = preprocessing.normalize([kw2])
    #     test_sim = np.full((test_num,know_num),0,float)
    #     for i in range(0,test_num):
    #         for j in range(0,know_num):
    #             if np.array_equal(k1[i,:], k2[j,:]) == True:
    #                 test_result[i] = know_leng
    #             else:    
    #                 for k in range(0, know_leng):
    #                     if k1[i,k] == k2[j,k]:
    #                         test_sim[i,j] = test_sim[i,j]+1
    #                 test_sim[i,j]= test_sim[i,j]/ know_leng*kw2[0][j]
    #         if test_result[i] == know_leng:
    #             pass
    #         else:        
    #             test_result[i] = sum(test_sim[i,:])
    #     #print(test_sim)
    #     print('test_result:', test_result) 
    #     return test_result


    # def knowledge_dis(self, k1, k2,kw1,kw2):
    #     k1_origion = k1.copy()
    #     k2_origion = k2.copy()
    #     know_length = np.shape(k1)[0]
    #     print('origioan k1 number:', know_length)
    #     know_length1 = np.shape(k2)[0]
    #     print('origioan k9 number:', know_length1)
    #     know_result = [0 for i in range (0, know_length)]
        
    #     for i in range(0,know_length):
    #         flag = 0
    #         j = 0
    #         while flag == 0:
    #             if np.array_equal(k1[i,:], k2[j,:]) == True:
    #                 know_result[i] = 1
    #                 flag = 1
    #                 k1_origion[i,:] = 0
    #                 k2_origion[j,:] = 0
    #                 kw1[i] = 0
    #                 kw2[j] = 0
    #             else:
    #                 j = j+1

    #             if j == know_length1:
    #                 flag = 1
    #                 know_result[i] = 0
    #     print('know_result:', know_result)
    #     #print(k1_origion)
    #     idx = np.argwhere(np.all(k1_origion[..., :] == 0, axis=1))
    #     k1_origion = np.delete(k1_origion, idx, axis=0)
    #     ## update weight
    #     kw1 = np.asarray(kw1)
    #     kw1 = kw1[kw1 != 0]
    #     idx = np.argwhere(np.all(k2_origion[..., :] == 0, axis=1))
    #     k2_origion = np.delete(k2_origion, idx, axis=0)  
    #     kw2 = np.asarray(kw2)
    #     kw2 = kw2[kw2 != 0]
        
    #     print(np.shape(k1_origion)[0],kw1) 
    #     print(np.shape(k2_origion)[0],kw2) 
    #     #print(k2_origion)    
    #     return k1_origion, k2_origion, kw1, kw2

def knowledge_weight_dict_gen(args, id_list, k_list):
    knowledge_weight_dict = dict()
    for id in id_list:
        knowledge_weight_dict[id] = dict()

    sub_events = list()
    for e in args.selected_events.split("_"):
        sub_events.extend(e.split("+"))
    for k_i, k in enumerate(k_list):
        if id_list[k_i] in sub_events:
            if str(k.astype(np.int32)) != '[0 0 0 0 0 0 0 0]':
                if not str(k.astype(np.int32)) in knowledge_weight_dict[id_list[k_i]].keys():
                    knowledge_weight_dict[id_list[k_i]][str(k.astype(np.int32))] = 1
                else:
                    knowledge_weight_dict[id_list[k_i]][str(k.astype(np.int32))] += 1
    # print(knowledge_weight_dict)
    return knowledge_weight_dict

def knowledge_weight_dict_dis(args, knowledge_weight_dict):
    knowledge_weight_dict_dis = dict()
    remove_list = list()
    for id1 in knowledge_weight_dict.keys():
        for id2 in knowledge_weight_dict.keys():
            if id1 != id2:
                for k1 in knowledge_weight_dict[id1].keys():
                    for k2 in knowledge_weight_dict[id1].keys():
                        if k1 == k2:
                            remove_list.append(k1)
    remove_list = list(set(remove_list))
    for id in knowledge_weight_dict.keys():
        knowledge_weight_dict_dis[id] = dict()
        for k in knowledge_weight_dict[id].keys():
            if k not in remove_list:
                knowledge_weight_dict_dis[id][k] = knowledge_weight_dict[id][k]
    return knowledge_weight_dict_dis

def knowledge_score_dict_gen(knowledge_weight_dict):
    knowledge_score_dict = dict()
    for id in knowledge_weight_dict.keys():
        knowledge_score_dict[id] = dict()
        for k in knowledge_weight_dict[id].keys():
            knowledge_score_dict[id][k] = knowledge_weight_dict[id][k] \
                                          / sum(knowledge_weight_dict[id].values())
    return knowledge_score_dict

def knowledge_weight_dict_infer(args, knowledge_weight_dict, knowledge_bit):
    predict_result = dict()
    for id in knowledge_weight_dict.keys():
        for k in knowledge_weight_dict[id].keys():
            if str(knowledge_bit.astype(np.int32)) == k:
                predict_result[id] = knowledge_weight_dict[id][k]

    knowledge_score_dict = knowledge_score_dict_gen(knowledge_weight_dict)

    if len(predict_result) == 1:
        predict_event = list(predict_result.keys())[0]
        # print("exactly match", predict_event)
    else:
        score_dict = dict()
        for id in knowledge_score_dict.keys():
            id_score = 0
            for k in knowledge_score_dict[id].keys():
                k_np = np.array([int(b) for b in k[1:-1].split(" ")])
                score = 0
                for bit_i, bit in enumerate(k_np):
                    if bit == knowledge_bit[bit_i]:
                        score += knowledge_score_dict[id][k] / len(k_np)
                id_score += score
            score_dict[id] = id_score

        max_score = max(score_dict.values())
        for id, score in score_dict.items():
            if score == max_score:
                predict_event = id

    if sorted(list(knowledge_weight_dict.keys())) == sorted(['1', '9', '10']):
        predict_event = predict_event
    else:
        if predict_event in ['1', '2', '8', '9', '10']:
            predict_event = '1+2+8+9+10'
        elif predict_event in ['3']:
            predict_event = '3'
        elif predict_event in ['4', '5']:
            predict_event = '4+5'
        elif predict_event in ['6', '7']:
            predict_event = '6+7'

    return predict_event
        
def knowledge_hist(args, id_list, n_list):
    # np.place(n_list, n_list==0, -1)
    # n_neg = np.where(n_list==0, 0, n_list)
    n_neg = np.where(n_list==0, -1, n_list)

    # for id_i, id in enumerate(id_list):
    #     # m = np.mean(n_neg[id_i])
    #     # mx = max(n_neg[id_i])
    #     # mn = min(n_neg[id_i])
    #     # n_neg[id_i] = (n_neg[id_i] -m) / (mx-mn)
    #     sum = np.sum(n_neg[id_i])
    #     n_neg[id_i] = n_neg[id_i]/sum

    knowledge_hist = dict()
    for id_i, id in enumerate(id_list):
        if id not in knowledge_hist.keys():
            knowledge_hist[id] = copy.deepcopy(n_neg[id_i].reshape((len(n_neg[id_i]),1)))
        else:
            knowledge_hist[id] += n_neg[id_i].reshape((len(n_neg[id_i]),1))

    value_list = list()
    for key, value in knowledge_hist.items():
        print(key, value.flatten())
        value_list.append(value.flatten())
        np.savetxt(cfg.code_path + "/data/" + args.exp + "/hist_raw.txt", np.array(value_list), fmt='%d')   

    # knowledge_hist = gen_knowledge_hist_nobias(knowledge_hist)
    # value_list = list()
    # for key, value in knowledge_hist.items():
    #     print(key, value.flatten())
    #     value_list.append(value.flatten())
    #     np.savetxt(cfg.code_path + "/output/hist_nobias.txt", np.array(value_list), fmt='%d')  

    # knowledge_hist_array = np.array(list(knowledge_hist.values()))
    # knowledge_hist_array_softmax = scipy.special.softmax(knowledge_hist_array, axis=0)
    # for i, (key, value) in enumerate(knowledge_hist.items()):
    #     knowledge_hist[key] = knowledge_hist_array_softmax[i]

    print("knowledge_hist:")
    for key, value in knowledge_hist.items():
        print(key, value.flatten())

    for id, hist in knowledge_hist.items():
        m = np.mean(hist)
        mx = max(hist)
        mn = min(hist)
        hist_pos = np.where(hist>0, hist, 0)
        hist_neg = np.where(hist<0, hist, 0)
        # hist_pos = hist[hist>0]
        # hist_neg = hist[hist<0]
        pos_max = np.sum(hist_pos)
        neg_max = -np.sum(hist_neg)

        # 加sigmoid

        hist_norm = np.zeros_like(hist)
        hist_norm = hist_norm + hist_pos / float(pos_max)
        hist_norm = hist_norm + hist_neg / float(neg_max)
        knowledge_hist[id] = hist_norm

    print("knowledge_hist:")
    for key, value in knowledge_hist.items():
        print(key, value.flatten())
    # print("knowledge_hist:")
        
    value_list = list()
    for key, value in knowledge_hist.items():
        print(key, value.flatten())
        value_list.append(value.flatten())
        np.savetxt(cfg.code_path + "/data/" + args.exp + "/hist_norm.txt", np.array(value_list), fmt='%f')   

        # plt.figure("hist")
        # fig, ax = plt.subplots(figsize=(10, 7))
        # ax.bar(
        #     x=list(range(len(value.flatten()))),  # Matplotlib自动将非数值变量转化为x轴坐标
        #     height=value,  # 柱子高度，y轴坐标
        #     width=0.6,  # 柱子宽度，默认0.8，两根柱子中心的距离默认为1.0
        #     align="center",  # 柱子的对齐方式，'center' or 'edge'
        #     color="red",  # 柱子颜色
        #     edgecolor="red",  # 柱子边框的颜色
        #     linewidth=2.0  # 柱子边框线的大小
        # )

        # # n, bins, patches = plt.hist(value.flatten(), facecolor='green', alpha=0.75)  
        # # plt.show()
        # plt.savefig("/Users/zzh/Code/SGF_v2/output/hist/" + str(key) + ".png", dpi=300)
    
    return knowledge_hist


def gen_knowledge_hist_nobias(knowledge_hist):
    knowledge_hist_nobias = dict()
    hist0 = list(knowledge_hist.values())[0]
    bias_list = list()
    for bit_i, bit in enumerate(hist0):
        min_bit = bit
        for id, hist in knowledge_hist.items():
            if hist[bit_i] <= min_bit:
                min_bit = hist[bit_i]
        bias_list.append(min_bit)
    bias_tensor = np.array(bias_list)
    for id, hist in knowledge_hist.items():
        knowledge_hist_nobias[id] = knowledge_hist[id] - bias_tensor

    return knowledge_hist_nobias


