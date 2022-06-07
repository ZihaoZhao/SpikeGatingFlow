from os import terminal_size
from dvsgesture_i import DatasetGesture_i
from expert import SGF_expert
import numpy as np
from STLayer import Spatiotemporal_Core
from logger import Logger
from agent import SGF_agent
import agent
from write_excel import write_excel
import cfg
from prior_knowledge import SGF_prior_knowledge
import knowledge
import train
import copy
from visualization_utils import save_visualize, save_curve, visualize, save_vis_formatted
from train import SGF_train

class SGF_inference(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp = args.exp
        self.save_excel = args.save_excel
        self.save_excel_path = args.save_excel_path
        self.selected_event = [event for event in args.selected_events.split("_")]
        self.sub_events = list()
        for e in self.args.selected_events.split("_"):
            self.sub_events.extend(e.split("+"))
        self.event_num =  len(self.selected_event)
        self.sub_event_num =  len(self.sub_events)
        self.st_paras = [int(st_para) for st_para in args.st_paras.split("_")]
        self.thres_s1 = [int(t) for t in args.thres_s1.split("_")]
        self.thres_s2 = [int(t) for t in args.thres_s2.split("_")]
        # self.thres_s3 = list()
        # self.thres_s3.append([int(t) for t in args.thres_s3.split("_")][0:2])
        # self.thres_s3.append([int(t) for t in args.thres_s3.split("_")][2:4])
        self.thres_s3 = [int(t) for t in args.thres_s3.split("_")]
        self.thres_s4 = [int(t) for t in args.thres_s4.split("_")]
        self.thres_s5 = [int(t) for t in args.thres_s5.split("_")]
        self.thres_bit_s1 = args.thres_bit_s1
        self.thres_bit_s2 = args.thres_bit_s2
        self.thres_bit_s3 = args.thres_bit_s3
        self.thres_bit_s4 = args.thres_bit_s4
        self.thres_bit_s5 = args.thres_bit_s5
        self.thres_step_s1 = args.thres_step_s1
        self.thres_step_s2 = args.thres_step_s2
        self.thres_step_s3 = args.thres_step_s3
        self.thres_step_s4 = args.thres_step_s4
        self.thres_step_s5 = args.thres_step_s5
        self.resolution_s1 = [int(t) for t in args.resolution_s1.split("_")]
        self.resolution_s2 = [int(t) for t in args.resolution_s2.split("_")]
        self.resolution_s3 = [int(t) for t in args.resolution_s3.split("_")]
        self.resolution_s4 = [int(t) for t in args.resolution_s4.split("_")]
        self.resolution_s5 = [int(t) for t in args.resolution_s5.split("_")]
        self.hopfield_frame_para = [int(para) for para in self.args.hopfield_frame_para.split("_")]
        self.code_mode = self.args.code_mode

        self.cnt_s3_2 = 0
        self.cnt_s3_10 = 0
        self.cnt_s3 = 0
        self.cnt_str = ""
        self.hop_hist_2 = np.zeros((1,4))
        self.hop_hist_10 = np.zeros((1,4))
        self.hop2_hist_dict = dict()
        for event in self.sub_events:
            self.hop2_hist_dict[event] = np.zeros((1,8))    

    def init_save_dict(self):
        self.cnt = 0                                                            ## total number 
        self.correct = 0                                                        ## correct one
        self.correct_dict       = dict()
        self.cnt_dict           = dict()
        self.missing_dict       = dict()
        self.incorrect_dict     = dict()
        self.unique_pre_dict    = dict()
        self.unique_cor_dict    = dict()
        self.knowledge_pre_dict = dict()
        self.knowledge_cor_dict = dict()
        self.expert1_cnt = 0
        self.train_fail_cnt = 0
        self.train_succ_cnt = 0

    def load_knowledge(self):
        self.unita_id = self.ReadTxtName(cfg.code_path + "/data/{:}/UnitA_id.txt".format(self.exp))
        self.unita_knowledge = np.loadtxt(cfg.code_path + "/data/{:}/UnitA_information.txt".format(self.exp))
        self.unitc_id = self.ReadTxtName(cfg.code_path + "/data/{:}/UnitC_id.txt".format(self.exp))
        self.unitc_knowledge = np.loadtxt(cfg.code_path + "/data/{:}/UnitC_information.txt".format(self.exp))

    def sgf_inference(self):
        dataset = DatasetGesture_i(cfg.data_path)    ## Read the inference dataset
        batchsize = 1
        testdata = np.full((80, 128, 128, 1),0)                       ## data initializations
        test_label = -1

        self.init_save_dict()
        self.load_knowledge()        
        for event in self.sub_events:
            self.cnt_dict[event] = 0     
            self.correct_dict[event] = 0   
            self.incorrect_dict[event] = 0    

        ##-----------------------------------------------------------##     
        ##                        Inference Phase                    ## 
        ##-----------------------------------------------------------## 
        for i in range(0, dataset.test_len(), batchsize):   
            # data preparing
            video, label = dataset.get_test_sample(i, reverse=False)
            if np.shape(video)[0] < 80:
                testdata[0:np.shape(video)[0],:,:,0] = video[:,:,:]
            else:    
                testdata[:,:,:,0] = video[0:80,:,:]
            test_label = label

            # SGF UnitA    class 1+2+8+9+10/3/4+5/6+7
            ##-----------------------------------------------------------##     
            ##        SGF UnitA feature vector score calculation         ## 
            ##-----------------------------------------------------------## 
            # ST core 1
            stlayer1 = Spatiotemporal_Core(testdata, 3, 2, 3, 2)         
            stlayer1.Spaceprocessing()
            stlayer1.Temporalprocessing()
            stlayer1.Stprocessing()
            stlayer1.stspike()
            e1 = SGF_expert(self.args)
            # Spatial SNN with feature index A and D
            space_neuron1 = e1.expert_space(self.resolution_s1[0], self.resolution_s1[1], \
                                stlayer1.ST_spike, self.thres_s1, [0,0], thres_bit=self.thres_bit_s1,  \
                                thres_step=self.args.thres_step_s1, thres_inc_factor=[-1,1], if_vote=self.args.vote_thres_step)
            # Spatial SNN with feature index B and C
            space_neuron2 = e1.expert_space(self.resolution_s2[0], self.resolution_s2[1], \
                                stlayer1.ST_spike, self.thres_s2, [0,0], thres_bit=self.thres_bit_s2, \
                                thres_step=self.args.thres_step_s2, thres_inc_factor=[1,-1], if_vote=self.args.vote_thres_step)
            space_all = np.concatenate((space_neuron1,space_neuron2), axis = 0)       ## jow results along rows, generate unique code          

            knowledge_weight_dict_a = knowledge.knowledge_weight_dict_gen(self.args, self.unita_id, self.unita_knowledge)
            predict_event_a = knowledge.knowledge_weight_dict_infer(self.args, knowledge_weight_dict_a, np.array(space_all))
            print(predict_event_a)

            testdata [testdata < 0] = 1              
            stlayer2 = Spatiotemporal_Core(testdata, 1, 1, 2, 2, if_st_neuron_clear=True)   
            stlayer2.Spaceprocessing()                                               ## space integration
            stlayer2.Temporalprocessing()                                            ## temporal integration
            stlayer2.Stprocessing()                                                  ## space-temporal integration
            stlayer2.stspike()                                                       ## generate acculumation spike
    
            # SGF UnitB    class 4/5 6/7
            ##-----------------------------------------------------------##     
            ##                SGF UnitB inference process                ## 
            ##-----------------------------------------------------------## 
            if predict_event_a in ["4+5", "6+7"]:
                final_results = np.full((1,2),0)
                for j in range(4,7): ## threshold
                    threshold = j*0.15 
                    for k in range (3,8):  ## frameskip
                        # Component E/F
                        area_index1, bit1,location_bit1, time1= e1.expert_hopfield(stlayer2.stcore[0:10*k:k,:,:,0],0,threshold) 
                        area_index3, bit3,location_bit3, time3= e1.expert_hopfield(stlayer2.stcore[0:10*k:k,:,:,0],1,threshold)
                        area_index2, bit2,location_bit2,time2= e1.expert_hopfield(stlayer2.stcore[0:10*k:k,:,:,0],2,threshold)
                        area_index4, bit4,location_bit4,time4= e1.expert_hopfield(stlayer2.stcore[0:10*k:k,:,:,0],3,threshold)
                        tagert13= np.stack((bit1,bit3),axis = 0)
                        tagert24= np.stack((bit2,bit4),axis = 0)
                        target1234 = np.concatenate((tagert13,tagert24),axis = 0)
                        location_bit = np.concatenate((location_bit1,location_bit2,location_bit3,location_bit4),axis = 0)
                        time_bit = [time1,time3,time2,time4] 
                        pk1 = SGF_prior_knowledge()
                        final = pk1.clockwise_knowledge(target1234,location_bit, time_bit)
                        if final[0] == 1:
                            final_results[0,0] = final_results[0,0] +1
                        elif final[1] == 1:  
                            final_results[0,1] = final_results[0,1] +1  
                print(final_results.flatten())    
                if final_results[0,0] > final_results[0,1]:
                    predict_event = predict_event_a.split("+")[1]
                else:
                    predict_event = predict_event_a.split("+")[0]


            # SGF UnitC    class 1/2/8/9/10
            ##-----------------------------------------------------------##     
            ##      SGF UnitC  feature vector score calculation          ## 
            ##-----------------------------------------------------------## 
            elif predict_event_a in ["1+2+8+9+10"]:
                # Spatial SNN with feature index G
                predict_event = e1.unitC_space_expert1(stlayer1.ST_spike)
                print(predict_event)
                if "+" in predict_event:
                    predict_event = e1.unitC_space_expert2(testdata)
                    print(predict_event)
                    if "+" in predict_event:
                        predict_event = e1.unitC_space_expert3(testdata)
                        print(predict_event)
                        if "+" in predict_event:
                            # Temporal SNNs with feature index H, I, J and K.
                            area_index1, ascent_bit1, swing_bit1= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,0:60,0], 0, 0.5, 14)
                            area_index3, ascent_bit3, swing_bit3= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,0:60,0], 1, 0.5, 14)                
                            area_index2, ascent_bit2, swing_bit2= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,60:120,0], 0, 0.5, 14)
                            area_index4, ascent_bit4, swing_bit4= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,60:120,0], 1, 0.5, 14)         
                            ascent_activites_td_left = ascent_bit1 + ascent_bit3  ## This for detection top->down and bottom-> up on the left area
                            ascent_activites_td_right = ascent_bit2 + ascent_bit4 ## This for detection top->down and bottom-> up on the right area
                            swing_activities_td_left = swing_bit1 + swing_bit3 
                            swing_activities_td_right = swing_bit2 + swing_bit4

                            area_index1, ascent_bit5, swing_bit5= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,0:60,0], 2, 0.5, 14)
                            area_index3, ascent_bit7, swing_bit7= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,0:60,0], 3, 0.5, 14)                
                            area_index2, ascent_bit6, swing_bit6= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,60:120,0], 2, 0.5, 14)
                            area_index4, ascent_bit8, swing_bit8= e1.expert_hopfield2(stlayer2.stcore[0:30:3,:,60:120,0], 3, 0.5, 14)      
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
                            # print(knowledge_bit)
                            knowledge_weight_dict_c = knowledge.knowledge_weight_dict_gen(self.args, self.unitc_id, self.unitc_knowledge)
                            predict_event_c = knowledge.knowledge_weight_dict_infer(self.args, knowledge_weight_dict_c, np.array(knowledge_bit))
                            # print(knowledge_weight_dict_c.keys())
                            # print(predict_event_c)
                            if predict_event_c == "9":
                                predict_event = "9"
                            else:
                                predict_event = "1"

            # Bypass   class 3
            else:
                predict_event = predict_event_a

            self.cnt += 1
            self.cnt_dict[str(test_label)] += 1
            if str(test_label) in predict_event:                       
                self.correct += 1        
                self.correct_dict[str(test_label)] += 1
                        
            info = "ID:{:} {:}/{:}  label:{:}  perdict:{:}".format(i, self.correct, self.cnt, test_label, predict_event)
            print(info)

        print("Total test sample number:",self.cnt)      
        print("the accurate rate:", self.correct/self.cnt) 
        for i in self.correct_dict.keys():  
            print("the",i,"event type accurate rate:", self.correct_dict[i]/self.cnt_dict[i])

    def ReadTxtName(self, rootdir):
        lines = []
        with open(rootdir, "r") as file_to_read:
            while True:
                line = file_to_read.readline()
                if not line:
                    break
                line = line.strip("\n")
                lines.append(line)
        return lines
