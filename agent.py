from expert import SGF_expert
from STLayer import Spatiotemporal_Core
import numpy as np
from Visualization import Visualization
from knowledge import Knowledge
from logger import Logger
import cfg
import os
from prior_knowledge import SGF_prior_knowledge

class SGF_agent(object):
    def __init__(self, args, train_dataset, label, exp, st_paras, train_succ_list):
        super().__init__()
        self.args = args
        self.resolution_s1 = [int(t) for t in args.resolution_s1.split('_')]
        self.resolution_s2 = [int(t) for t in args.resolution_s2.split('_')]
        self.resolution_s3 = [int(t) for t in args.resolution_s3.split('_')]
        self.thres_bit_s1 = args.thres_bit_s1
        self.thres_bit_s2 = args.thres_bit_s2
        self.selected_event = [event for event in self.args.selected_events.split('_')]
        self.sub_events = list()
        for e in self.args.selected_events.split("_"):
            self.sub_events.extend(e.split("+"))
        self.hopfield_frame_para = [int(para) for para in self.args.hopfield_frame_para.split('_')]
        self.code_mode = self.args.code_mode
        self.event_num =  len(self.selected_event)
        self.sub_event_num =  len(self.sub_events)
        self.train_data = train_dataset
        self.exp = exp
        self.label = label
        self.st_paras = st_paras
        self.train_succ_list = train_succ_list

    def get_train_succ_list(self):
        return self.train_succ_list

    def agent_binary_tree(self, batch_i, data, thres_s1, thres_s2, thres_s3, offset, 
                                                    expert1_id, expert1_knowledge):
        
        # print("Space expert 1 domain:")
        log_filename = cfg.code_path + "/data/" + self.exp + "/train_result.log"
        if batch_i == 0 and os.path.exists(log_filename):
            os.remove(log_filename)
        logger_object = Logger(log_filename)
        e1 = SGF_expert(self.args)                                                            ## experts initializations
        k1 = Knowledge(self.args)                                                             ## knolwledge initializations
        # Spatial SNN with feature index A and D
        space_neuron1 = e1.expert_space(self.resolution_s1[0], self.resolution_s1[1], \
                                            data, thres_s1, offset, thres_bit=self.thres_bit_s1,  \
                                            thres_step=self.args.thres_step_s1, thres_inc_factor=[-1,1], if_vote=self.args.vote_thres_step)    
        # Spatial SNN with feature index B and C               
        space_neuron2 = e1.expert_space(self.resolution_s2[0], self.resolution_s2[1],  \
                                            data, thres_s2, offset, thres_bit=self.thres_bit_s2,  \
                                            thres_step=self.args.thres_step_s2, thres_inc_factor=[1,-1], if_vote=self.args.vote_thres_step)                ## space expert 2 computing (col, row)
        space_all = np.concatenate((space_neuron1, space_neuron2), axis = 0)      ## combine two experts knowledge

        similarity = Knowledge(self.args).check_similarity(space_all, self.label, self.args.selected_events.split("_"))
        np.fill_diagonal(similarity, 1)  ## fill diagnoal to 1
        n1, id1 = k1.knowledge_base(space_all, self.label, expert1_id, expert1_knowledge)   ## generate expert knowledge   
        if np.all(similarity) == 1:
            print('training sucessful')
            logger_object.info(str(batch_i)+ ' training successful')    
            self.train_succ_list.append(1)  
            print("UnitA new knowledge:")
            for id_i, id in enumerate(self.label):
                print(self.label[id_i], space_all[:, id_i])                                                 
        else:
            print('training failed')
            logger_object.info(str(batch_i)+ ' training failed')     
            self.train_succ_list.append(0)                                 

        if len(id1) != 1 and id1[0] == '-1':
            n1 = n1[1:, :]
            id1 = id1[1:]

        return n1, id1

    def check_knowledge(self, id_list, knowledge_list, forbidden_id_list):
        id_list_new = list()
        knowledge_list_new = list()
        for id_i, id in enumerate(id_list):
            if id not in forbidden_id_list:
                id_list_new.append(id_list[id_i])
                knowledge_list_new.append(knowledge_list[id_i])

        if len(id_list_new) == 0:
            id_list_new = [-1]
            knowledge_list_new = -1*np.ones((1, len(knowledge_list[0,:])))

        return id_list_new, np.array(knowledge_list_new)


    # def check_similarity(self, data):
    #     if data.shape[0] == data.size:
    #         data = data.reshape(data.shape[0], 1)
    #     similarity = np.full((np.shape(data)[1],np.shape(data)[1]),0)
    #     for i in range(0, np.shape(data)[1]):   ## event
    #        for j in range(0,np.shape(data)[1]): ## event
    #            if np.all(data[:,i] == data[:,j]): 
    #                similarity[i,j] = 0
    #            else:
    #                similarity[i,j] = 1    
    #     return similarity 

def find_movement_index(data): 
        max_value = max(data) 
        data = data.tolist()
        max_index = data.index(max_value)
        '''
        if len(max_index) == 1:  ## if there is only one maximum value
            movement_index = max_index
        else:
            movement_index =[] 
        '''       
        return max_index 
