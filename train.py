import os
from visualization_utils import save_visualize, save_curve, visualize, save_vis_formatted
import numpy as np
from dvsgesture_t import DatasetGesture
from Visualization import Visualization
from SensingLayer import SensingLayer
from STLayer import Spatiotemporal_Core
from agent import SGF_agent
from expert import SGF_expert
import cfg
import copy
import random
from prior_knowledge import SGF_prior_knowledge
import knowledge
# from logger import Logger

class SGF_train(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.selected_event = [event for event in args.selected_events.split('_')]
        self.sub_events = list()
        for e in self.args.selected_events.split("_"):
            self.sub_events.extend(e.split("+"))
        self.event_num =  len(self.selected_event)
        self.sub_event_num =  len(self.sub_events)
        self.st_paras = [int(st_para) for st_para in args.st_paras.split('_')]
        self.thres_s1 = [int(t) for t in args.thres_s1.split('_')]
        self.thres_s2 = [int(t) for t in args.thres_s2.split('_')]
        self.thres_s3 = list()
        self.thres_s3.append([int(t) for t in args.thres_s3.split('_')][0:2])
        self.thres_s3.append([int(t) for t in args.thres_s3.split('_')][2:4])
        self.thres_bit_s1 = args.thres_bit_s1
        self.thres_bit_s2 = args.thres_bit_s2
        self.resolution_s1 = [int(t) for t in args.resolution_s1.split('_')]
        self.resolution_s2 = [int(t) for t in args.resolution_s2.split('_')]
        self.resolution_s3 = [int(t) for t in args.resolution_s3.split('_')]
        self.exp = args.exp
        self.exp_dir = cfg.code_path + "/data/" + self.exp
        self.save_st_sore = args.save_st_core
        self.train_succ_list = list()
        self.train_succ_cnt = 0
        self.train_fail_cnt = 0
        self.code_mode = self.args.code_mode
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

    def save_knowledge(self, expert_name, id, knowledge, if_save=True):
        if (not isinstance(id,int)):
            # print("expert2:")
            sort_idx = sorted(range(len(id)), key=lambda k: id[k])
            sorted_knowledge = list(map(lambda x:knowledge[x], sort_idx))
            sorted_id = sorted(id)
            for event_i, event in enumerate(id):
                if not if_save:
                    print(event, knowledge[event_i], sorted_id[event_i], sorted_knowledge[event_i])   
            expert_id = cfg.code_path + "/data/" + self.exp + "/"+expert_name+"_id.txt"
            expert_information = cfg.code_path + "/data/" + self.exp + "/"+expert_name+"_information.txt"
            if (isinstance(knowledge, list)):
                len_n = len(knowledge)
                n = np.array(knowledge).reshape(1, len_n)
            if if_save:
                np.savetxt(expert_id, sorted_id, fmt='%s')
                np.savetxt(expert_information, sorted_knowledge, fmt='%d')   

    def sgf_train(self, data_num, iter):
        # data:  training data
        # times: training times
        # random seed
        dataset = DatasetGesture(cfg.data_path)            ## Read training data

        ## select event
        selected_event = self.sub_events
        print(self.args.selected_events)
        print(selected_event)
        # 1: hand clip
        # 2: left wave
        # 3: right wave
        # 4: left counter clock-wise
        # 5: left clock-wise
        # 6: right counter clock-wise
        # 7: right clock-wise
        # 8: arm roll
        # 9: arm drum
        # 10: air guitar
        # 11: random

        train_data_folder = os.path.join(cfg.data_path, 'train_npy')
        train_filenames_all = os.listdir(train_data_folder)

        random.seed(0)
        hop2_knowledge = list()
        hop2_id = list()
        all_data_list = range(0, dataset.train_num)
        assert data_num <= dataset.train_num
        selected_sample = random.sample(all_data_list, data_num)
        print(selected_sample)

        ##-----------------------------------------------------------##     
        ##                        Data preprocessing                 ## 
        ##-----------------------------------------------------------## 
        print("Preparing training data...")
        event_mix = False
        train_filenames = list()
        for filename in train_filenames_all:
            for event in selected_event:
                if not "+" in event:
                    event_mix = True
                    for sample in selected_sample:
                        match_str = "train_" + str(event) + "_" + str(sample) + ".npy"
                        if match_str in filename:
                            train_filenames.append(filename)
                else:
                    event_list = event.split("+")
                    for event in event_list:
                        for sample in selected_sample:
                            match_str = "train_" + str(event) + "_" + str(sample) + ".npy"
                            if match_str in filename:
                                train_filenames.append(filename)

        if self.args.test_batch_list == '0':
            test_batch_list = list()
        else:
            test_batch_list = [int(t) for t in self.args.test_batch_list.split('_')]



        ##-----------------------------------------------------------##     
        ##                         Training Phase                    ## 
        ##-----------------------------------------------------------## 
        for i in range(0, iter):
            print("----------------------------------------------")
            print(self.exp, ", start the", i, "/", iter, "training iteration:")

            if self.args.each_sample_train_once:
                assert (iter == data_num)
                selected_batch_sample = selected_sample[i]
            else:
                if self.args.inner_batch_random:
                    selected_batch_sample = random.sample(selected_sample, self.sub_event_num)
                else:
                    if self.args.test_batch_list == '0':
                        selected_batch_sample = random.sample(selected_sample, 1)[0]
                    else:
                        selected_batch_sample = test_batch_list[i]
            if self.args.test_batch_list == '0':
                test_batch_list.append(selected_batch_sample)
            batch_filenames = list()
            cut_frame = 80
            batch_data = np.full((cut_frame, 128, 128, self.sub_event_num), 0)

            for filename in train_filenames:
                for event_i, event in enumerate(selected_event):
                    if self.args.inner_batch_random:
                        match_str = "train_" + str(event) + "_" + str(selected_batch_sample[event_i]) + ".npy"
                    else:
                        match_str = "train_" + str(event) + "_" + str(selected_batch_sample) + ".npy"
                    if match_str in filename:
                        # print(match_str, filename)
                        batch_filenames.append(filename)      
            batch_filenames.sort()
            
            # load np data and trancate 80 frame
            for filename in batch_filenames:
                np_name = os.path.join(train_data_folder, filename)
                # print(np_name)
                sample = np.load(np_name)
                event = str(filename.split("_")[-2])
                event_i = selected_event.index(event)
                if np.shape(sample)[0] >= cut_frame:
                    batch_data[:, :, :, event_i] =  sample[0:cut_frame, :, :] 
                else:
                    batch_data[0:np.shape(sample)[0], :, :, event_i] =  sample   


            e1 = SGF_expert(self.args)
            stlayer1 = Spatiotemporal_Core(batch_data, 3, 2, 3, 2)        ## ST layer processing      
            stlayer1.Spaceprocessing()                                               ## space integration
            stlayer1.Temporalprocessing()                                            ## temporal integration
            stlayer1.Stprocessing()                                                  ## space-temporal integration
            stlayer1.stspike()                                                       ## generate acculumation spike

            testdata = copy.deepcopy(batch_data)
            testdata [testdata < 0] = 1                                                      ## generate acculumation spike
            
            stlayer2 = Spatiotemporal_Core(testdata[:,:,:,2:5], 1, 1, 2, 2, if_st_neuron_clear=True)   
            stlayer2.Spaceprocessing()                                               ## space integration
            stlayer2.Temporalprocessing()                                            ## temporal integration
            stlayer2.Stprocessing()                                                  ## space-temporal integration
            stlayer2.stspike()                                                       ## generate acculumation spike
            
            ##-----------------------------------------------------------##     
            ##                          SGF Unit A                       ## 
            ##-----------------------------------------------------------## 
            a1 = SGF_agent(self.args, batch_data, selected_event, exp=self.exp, st_paras=self.st_paras, train_succ_list=self.train_succ_list)       

            if i == 0:                                                              ## expert knowledge initializations
                event_pre_id = [-1]   # pre id
                k1_len = self.resolution_s1[0]*self.resolution_s1[1]*self.thres_bit_s1*self.thres_bit_s1 \
                            + self.resolution_s2[0]*self.resolution_s2[1]*self.thres_bit_s2*self.thres_bit_s2
                expert1_pre_knowledge = np.array([-1 for x in range(k1_len)])                      ## user defined at this version                       
            else:
                event_pre_id = id1
                expert1_pre_knowledge = n1

            # generate feature vectors
            n1, id1 = a1.agent_binary_tree(i, stlayer1.ST_spike, \
                                        thres_s1=self.thres_s1, \
                                        thres_s2=self.thres_s2, \
                                        thres_s3=self.thres_s3, \
                                        offset=[0,0] , \
                                        expert1_id=event_pre_id, expert1_knowledge=expert1_pre_knowledge)  ## agent binary_tree_search_policy


            # SGF Unit B doesn't require trainings since it is designed based on the human prior knowledge.

            ##-----------------------------------------------------------##     
            ##                          SGF Unit C                       ## 
            ##-----------------------------------------------------------## 
            frame_skip = self.args.hop2_frame_skip          #3
            threshold = self.args.hop2_threshold            #0.2
            hist_threshold = self.args.hop2_hist_threshold  #15
            print("UnitC new knowledge:")
            for l in range(stlayer2.stcore.shape[3]):
                # Temporal SNNs with feature index H, I, J and K.
                area_index1, ascent_bit1, swing_bit1= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,0:60,l],0,threshold,hist_threshold)
                area_index3, ascent_bit3, swing_bit3= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,0:60,l],1,threshold,hist_threshold)                
                area_index2, ascent_bit2, swing_bit2= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,60:120,l],0,threshold,hist_threshold)
                area_index4, ascent_bit4, swing_bit4= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,60:120,l],1,threshold,hist_threshold)                
                ascent_activites_td_left = ascent_bit1 + ascent_bit3  ## This for detection top->down and bottom-> up on the left area
                ascent_activites_td_right = ascent_bit2 + ascent_bit4 ## This for detection top->down and bottom-> up on the right area
                swing_activities_td_left = swing_bit1 + swing_bit3 
                swing_activities_td_right = swing_bit2 + swing_bit4
                
                area_index1, ascent_bit5, swing_bit5= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,0:60,l],2,threshold,hist_threshold)
                area_index3, ascent_bit7, swing_bit7= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,0:60,l],3,threshold,hist_threshold)                
                area_index2, ascent_bit6, swing_bit6= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,60:120,l],2,threshold,hist_threshold)
                area_index4, ascent_bit8, swing_bit8= e1.expert_hopfield2(stlayer2.stcore[0:10*frame_skip:frame_skip,:,60:120,l],3,threshold,hist_threshold)                
                ascent_activites_lr_left = ascent_bit5   + ascent_bit7 
                ascent_activites_lr_right = ascent_bit6  + ascent_bit8 
                swing_activities_lr_left = swing_bit5 + swing_bit7 
                swing_activities_lr_right = swing_bit6 + swing_bit8
                
                feature_bit  = [np.where(ascent_activites_td_left+ascent_activites_td_right>0,1,0) , 
                                np.where(swing_activities_td_left+swing_activities_td_right>0,1,0) , 
                                np.where(ascent_activites_lr_left +swing_activities_lr_left>0,1,0) , 
                                np.where( ascent_activites_lr_right + swing_activities_lr_right>0,1,0) ]
                location_bit = [np.where(ascent_activites_td_left + ascent_activites_lr_left>0,1,0) , 
                                np.where(swing_activities_td_left+swing_activities_lr_left>0,1,0) , 
                                np.where(ascent_activites_td_right+ swing_activities_td_right>0,1,0) ,
                                np.where( ascent_activites_lr_right+swing_activities_lr_right>0,1,0) ]
                knowledge_bit = np.concatenate((feature_bit,location_bit),axis = 0)
                hop2_knowledge.append(knowledge_bit)
                hop2_id.append(selected_event[2+l])
                print(selected_event[2+l], knowledge_bit)   

        # save feature vector
        self.save_knowledge("UnitA", id1, n1, if_save=True) 
        self.save_knowledge("UnitC", hop2_id, np.array(hop2_knowledge), if_save=True) 