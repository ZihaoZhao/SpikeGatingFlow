from dvsgesture_i import DatasetGesture_i
from expert import SGF_expert
import numpy as np
from STLayer import Spatiotemporal_Core
from logger import Logger
from agent import SGF_agent
import cfg

class SGF_profiling(object):
    def __init__(self):
        super().__init__()


    def dataset_profiling(self):  
        name = "dataset_profiling"
        logger_object = Logger(name)
        logger_object_0 = Logger('event_0')
        logger_object_1 = Logger('event_1')
        logger_object_2 = Logger('event_2')
        logger_object_3 = Logger('event_3') 
        logger_object_4 = Logger('event_4')
        logger_object_5 = Logger('event_5')
        logger_object_6 = Logger('event_6')
        logger_object_7 = Logger('event_7')
        logger_object_8 = Logger('event_8')
        logger_object_9 = Logger('event_9') 

        sample_code_0 = np.full((1,46),0) 
        sample_code_1 = np.full((1,46),0)         
        sample_code_2 = np.full((1,46),0) 
        sample_code_3 = np.full((1,46),0) 
        sample_code_4 = np.full((1,46),0)         
        sample_code_5 = np.full((1,46),0)
        sample_code_6 = np.full((1,46),0) 
        sample_code_7 = np.full((1,46),0) 
        sample_code_8 = np.full((1,46),0)         
        sample_code_9 = np.full((1,46),0)

        pattern0 =  [0 for i in range (500)]
        pattern1 =  [0 for i in range (500)]              ## 30 is the user defined type for a single event 
        pattern2 =  [0 for i in range (500)]
        pattern3 =  [0 for i in range (500)]
        pattern4 =  [0 for i in range (500)]
        pattern5 =  [0 for i in range (500)]              ## 30 is the user defined type for a single event 
        pattern6 =  [0 for i in range (500)]
        pattern7 =  [0 for i in range (500)]
        pattern8 =  [0 for i in range (500)]
        pattern9 =  [0 for i in range (500)]   

        num_his_9 = 0 
        num_his_8 = 0 
        num_his_7 = 0 
        num_his_6 = 0                
        num_his_5 = 0 
        num_his_4 = 0 
        num_his_3 = 0 
        num_his_2 = 0  
        num_his_1 = 0 
        num_his_0 = 0                  


        dataset = DatasetGesture_i(cfg.data_path)    
        batchsize = 10
        cnt = 0
        testdata = np.full((80,128,128,batchsize),0)                       ## data initializations
        test_label = [0 for i in range (batchsize)]
        for i in range(0,dataset.test_len(),batchsize):      ## read dataset by a block 10
            for j in range(0, batchsize):
                index = j+ i
                if index >= dataset.test_len():
                    index = dataset.test_len()-1
                else:
                    index = index
                video, label = dataset.get_test_sample(index)
                if np.shape(video)[0] < 80:
                    testdata[0:np.shape(video)[0],:,:,j] = video[:,:,:]
                else:    
                    testdata[:,:,:,j] = video[0:80,:,:]
                test_label[j] = label

            stlayer = Spatiotemporal_Core(testdata, 2, 1, 2, 1)         
            stlayer.Spaceprocessing()
            stlayer.Temporalprocessing()
            stlayer.Stprocessing()
            stlayer.stspike()
            e1 = SGF_expert()
            space_neuron = e1.expert_space( 2, stlayer.ST_spike, [20,10], [0,0])
            space_neuron1 = e1.expert_space( 2, stlayer.ST_spike, [5,200], [0,0])
            space_all = np.concatenate((space_neuron,space_neuron1), axis = 0)       ## jow results along rows, generate unique code                              
            data_ds = testdata[0:80:5,:,:,:]
            #data_ds[data_ds < 0] = 0                            
            stlayer = Spatiotemporal_Core(data_ds, 2, 1, 2, 1)                     ##: user defined (this should done by agent)  
            stlayer.Spaceprocessing()                                              ## space integration
            stlayer.Temporalprocessing()                                           ## temporal integration
            stlayer.Stprocessing()                                                 ## space-temporal integration
            stlayer.stspike() 
            if np.any(space_neuron1[2,:] == 1):    ## simple attention mechanisms
                start = [0,0]
                end = [64,40]
            elif np.any(space_neuron1[3,:] == 1):
                start = [0,20]  
                end = [64,64]
            else:
                start = [0,0]  
                end = [64,64]                      
            temporal_neuron_final1 = e1.expert_temporal(2, stlayer.stcore, start, end, 0, int(50/64*stlayer.stcore.shape[1])) #--: expert_temporal(self, resolution, data, start, end, type, scale):
            temporal_neuron_final2 = e1.expert_temporal(2, stlayer.stcore, start, end ,1, int(50/64*stlayer.stcore.shape[1])) #--: expert_temporal(self, resolution, data, start, end, type, scale):
            temporal_neuron1 =  [0 for i in range (len(temporal_neuron_final1))]
            temporal_neuron2 =  [0 for i in range (len(temporal_neuron_final2))]                  
            for i in range(0, len(temporal_neuron_final1)):
                if temporal_neuron_final1[i] > temporal_neuron_final2[i]:
                    temporal_neuron1[i] = 1
                else:
                    temporal_neuron2[i] = 1                     
            temporal_td = np.stack((temporal_neuron1,temporal_neuron2),axis = 0)
            temporal_all = np.concatenate((space_all,temporal_td),axis = 0)
            space_neuron3 = e1.expert_space(6, stlayer.ST_spike, [8,5], [0,0])
            sp_all= np.concatenate((temporal_all,space_neuron3), axis = 0)      ## combine two experts knowledge  
                

            for i in range(0, batchsize): 
                    sample = sp_all[:,i] 
                    file_name_id = dataset.get_test_data_file_name(cnt)
                    print(cnt,file_name_id)
                    if test_label[i] == 9:   ## event is 9 
                        pattern_num = np.shape(sample_code_9)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_9[j]):   ## if it is the same pattern, record the number
                                    pattern9[j] =  pattern9[j] +1
                                    pattern_index[j] = 1
                                    logger_object_9.debug(pattern9)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_9 = np.vstack([sample_code_9,sp_all[:,i]]) 
                                logger_object_9.critial(sample_code_9)  
                                logger_object_9.warn(file_name_id) 
                                logger_object_9.debug(cnt) 

                    num_his_9 = self.expert_profiling(sample_code_9, num_his_9)


                    if test_label[i] == 0:   ## event is 0 
                        pattern_num = np.shape(sample_code_0)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_0[j]):   ## if it is the same pattern, record the number
                                    pattern0[j] =  pattern0[j] +1
                                    pattern_index[j] = 1
                                    logger_object_0.debug(pattern0)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_0 = np.vstack([sample_code_0,sp_all[:,i]]) 
                                logger_object_0.critial(sample_code_0)  
                                logger_object_0.warn(file_name_id) 
                                logger_object_0.debug(cnt)
                    num_his_0 = self.expert_profiling(sample_code_0, num_his_0)                                

                    if test_label[i] == 1:   ## event is 1 
                        pattern_num = np.shape(sample_code_1)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_1[j]):   ## if it is the same pattern, record the number
                                    pattern1[j] =  pattern1[j] +1
                                    pattern_index[j] = 1
                                    logger_object_1.debug(pattern1)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_1 = np.vstack([sample_code_1,sp_all[:,i]]) 
                                logger_object_1.critial(sample_code_1)  
                                logger_object_1.warn(file_name_id) 
                                logger_object_1.debug(cnt) 
                    num_his_1 = self.expert_profiling(sample_code_1, num_his_1)                                 

                    if test_label[i] == 2:   ## event is 2
                        pattern_num = np.shape(sample_code_2)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_2[j]):   ## if it is the same pattern, record the number
                                    pattern2[j] =  pattern2[j] +1
                                    pattern_index[j] = 1
                                    logger_object_2.debug(pattern2)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_2 = np.vstack([sample_code_2,sp_all[:,i]]) 
                                logger_object_2.critial(sample_code_2)  
                                logger_object_2.warn(file_name_id) 
                                logger_object_2.debug(cnt) 
                    num_his_2 = self.expert_profiling(sample_code_2, num_his_2)                                   

                    if test_label[i] == 3:   ## event is 3 
                        pattern_num = np.shape(sample_code_3)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_3[j]):   ## if it is the same pattern, record the number
                                    pattern3[j] =  pattern3[j] +1
                                    pattern_index[j] = 1
                                    logger_object_3.debug(pattern3)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_3 = np.vstack([sample_code_3,sp_all[:,i]]) 
                                logger_object_3.critial(sample_code_3)  
                                logger_object_3.warn(file_name_id) 
                                logger_object_3.debug(cnt) 
                    num_his_3 = self.expert_profiling(sample_code_3, num_his_3)                                   

                    if test_label[i] == 4:   ## event is 4 
                        pattern_num = np.shape(sample_code_4)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_4[j]):   ## if it is the same pattern, record the number
                                    pattern4[j] =  pattern4[j] +1
                                    pattern_index[j] = 1
                                    logger_object_4.debug(pattern4)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_4 = np.vstack([sample_code_4,sp_all[:,i]]) 
                                logger_object_4.critial(sample_code_4)  
                                logger_object_4.warn(file_name_id) 
                                logger_object_4.debug(cnt) 
                    num_his_4 = self.expert_profiling(sample_code_4, num_his_4)                                    

                    if test_label[i] == 5:   ## event is 5 
                        pattern_num = np.shape(sample_code_5)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_5[j]):   ## if it is the same pattern, record the number
                                    pattern5[j] =  pattern5[j] +1
                                    pattern_index[j] = 1
                                    logger_object_5.debug(pattern5)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_5 = np.vstack([sample_code_5,sp_all[:,i]]) 
                                logger_object_5.critial(sample_code_5)  
                                logger_object_5.warn(file_name_id) 
                                logger_object_5.debug(cnt) 
                    num_his_5 = self.expert_profiling(sample_code_5, num_his_5)                                    

                    if test_label[i] == 6:   ## event is 6 
                        pattern_num = np.shape(sample_code_6)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_6[j]):   ## if it is the same pattern, record the number
                                    pattern6[j] =  pattern6[j] +1
                                    pattern_index[j] = 1
                                    logger_object_6.debug(pattern6)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_6 = np.vstack([sample_code_6,sp_all[:,i]]) 
                                logger_object_6.critial(sample_code_6)  
                                logger_object_6.warn(file_name_id) 
                                logger_object_6.debug(cnt)    

                    if test_label[i] == 7:   ## event is 7 
                        pattern_num = np.shape(sample_code_7)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_7[j]):   ## if it is the same pattern, record the number
                                    pattern7[j] =  pattern7[j] +1
                                    pattern_index[j] = 1
                                    logger_object_7.debug(pattern7)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_7 = np.vstack([sample_code_7,sp_all[:,i]]) 
                                logger_object_7.critial(sample_code_7)  
                                logger_object_7.warn(file_name_id) 
                                logger_object_7.debug(cnt)

                    if test_label[i] == 8:   ## event is 8 
                        pattern_num = np.shape(sample_code_8)[0]          ## calculate the current pattern num
                        pattern_index = [0 for i in range (pattern_num)]  
                        for j in range(0, pattern_num):
                                if np.all(sample  == sample_code_8[j]):   ## if it is the same pattern, record the number
                                    pattern8[j] =  pattern8[j] +1
                                    pattern_index[j] = 1
                                    logger_object_8.debug(pattern8)  
                        if np.all(np.asarray(pattern_index) == 0):
                                sample_code_8 = np.vstack([sample_code_8,sp_all[:,i]]) 
                                logger_object_8.critial(sample_code_8)  
                                logger_object_8.warn(file_name_id) 
                                logger_object_8.debug(cnt)
                    cnt +=1



        logger_object_9.debug(np.shape(sample_code_9)[0]-1) 
        logger_object_9.debug(pattern9)
        event9_id_p = "expert4_id_p"
        np.save(event9_id_p, pattern9)
        event9_pattern_p = 'event9_pattern_p'
        np.save(event9_pattern_p, sample_code_9) 
        event9_expert_p = "event9_expert_p"
        np.save(event9_expert_p, num_his_9)

        logger_object_8.debug(np.shape(sample_code_8)[0]-1) 
        logger_object_8.debug(pattern8)
        event8_id_p = "expert8_id_p"
        np.save(event8_id_p, pattern8)
        event8_pattern_p = 'event8_pattern_p'
        np.save(event8_pattern_p, sample_code_8)          

        logger_object_7.debug(np.shape(sample_code_7)[0]-1) 
        logger_object_7.debug(pattern7)
        event7_id_p = "expert7_id_p"
        np.save(event7_id_p, pattern7)
        event7_pattern_p = 'event7_pattern_p'
        np.save(event7_pattern_p, sample_code_7)              

        logger_object_6.debug(np.shape(sample_code_6)[0]-1) 
        logger_object_6.debug(pattern6)  
        event6_id_p = "expert6_id_p"
        np.save(event6_id_p, pattern6)
        event6_pattern_p = 'event6_pattern_p'
        np.save(event6_pattern_p, sample_code_6)  

        logger_object_5.debug(np.shape(sample_code_5)[0]-1) 
        logger_object_5.debug(pattern5)
        event5_id_p = "expert5_id_p"
        np.save(event5_id_p, pattern5)
        event5_pattern_p = 'event5_pattern_p'
        np.save(event5_pattern_p, sample_code_5) 
        event5_expert_p = "event5_expert_p"
        np.save(event5_expert_p, num_his_5)                      

        logger_object_4.debug(np.shape(sample_code_4)[0]-1) 
        logger_object_4.debug(pattern4)
        event4_id_p = "expert4_id_p"
        np.save(event4_id_p, pattern4)
        event4_pattern_p = 'event4_pattern_p'
        np.save(event4_pattern_p, sample_code_4)
        event4_expert_p = "event4_expert_p"
        np.save(event4_expert_p, num_his_4)                      

        logger_object_3.debug(pattern3)                      
        logger_object_3.debug(np.shape(sample_code_3)[0]-1)
        event3_id_p = "expert3_id_p"
        np.save(event3_id_p, pattern3)
        event3_pattern_p = 'event3_pattern_p'
        np.save(event3_pattern_p, sample_code_3)
        event3_expert_p = "event3_expert_p"
        np.save(event3_expert_p, num_his_3)                      

        logger_object_2.debug(pattern2)         
        logger_object_2.debug(np.shape(sample_code_2)[0]-1)
        event2_id_p = "expert2_id_p"
        np.save(event2_id_p, pattern2)
        event2_pattern_p = 'event2_pattern_p'
        np.save(event2_pattern_p, sample_code_2)
        event2_expert_p = "event2_expert_p"
        np.save(event2_expert_p, num_his_2)                      

        logger_object_1.debug(pattern1)                      
        logger_object_1.debug(np.shape(sample_code_1)[0]-1) 
        event1_id_p = "expert1_id_p"
        np.save(event1_id_p, pattern1)
        event1_pattern_p = 'event1_pattern_p'
        np.save(event1_pattern_p, sample_code_1)
        event1_expert_p = "event1_expert_p"
        np.save(event1_expert_p, num_his_1)                      

        logger_object_0.debug(pattern0)         
        logger_object_0.debug(np.shape(sample_code_0)[0]-1)  
        event0_id_p = "expert0_id_p"
        np.save(event0_id_p, pattern0)
        event0_pattern_p = 'event0_pattern_p'
        np.save(event0_pattern_p, sample_code_0) 
        event0_expert_p = "event0_expert_p"
        np.save(event0_expert_p, num_his_0)            


    def expert_profiling(self,sample_code,num_his):  
        num = np.shape(sample_code)[0]-1 
        num_his = np.vstack([num_his,num])
        return  num_his

