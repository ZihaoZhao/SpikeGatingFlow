import numpy as np
from Neuron import Neuron
import cfg

class Spatiotemporal_Core(Neuron):
    def __init__(self,  dataset, 
                        c_space_num, s_thre, t_window, t_thre, if_st_neuron_clear=False):
        super().__init__()           
        self.c_space_num = c_space_num
        self.s_thre = s_thre
        self.t_window = t_window
        self.t_thre = t_thre
        self.dataset = dataset
        self.if_st_neuron_clear = if_st_neuron_clear
            
    def Spaceprocessing(self): 
        neuron1 = Neuron(self.if_st_neuron_clear)      
        #self.space= np.full((np.shape(self.dataset)[0], int(np.shape(self.dataset)[1]/self.c_space_num), int(np.shape(self.dataset)[2]/self.c_space_num), np.shape(self.dataset)[3]),0)
        init= np.full((int(np.shape(self.dataset)[1]/self.c_space_num), int(np.shape(self.dataset)[2]/self.c_space_num), np.shape(self.dataset)[3]),0)
        self.V_space= np.full((np.shape(self.dataset)[0], int(np.shape(self.dataset)[1]/self.c_space_num), int(np.shape(self.dataset)[2]/self.c_space_num), np.shape(self.dataset)[3]),0)
        self.spaceneuron= np.full((np.shape(self.dataset)[0], int(np.shape(self.dataset)[1]/self.c_space_num), int(np.shape(self.dataset)[2]/self.c_space_num), np.shape(self.dataset)[3]),0)
        for i in range(0, np.shape(self.dataset)[3]):  ##event
           for j in range(0, np.shape(self.dataset)[0]):  ##frame
              for k in range(0, np.shape(self.spaceneuron)[1]):
                  for l in range(0, np.shape(self.spaceneuron)[2]):
                     # self.space[j,k,l,i] = np.where(sum(sum(self.dataset[j, k:k+self.c_space_num, l:l+self.c_space_num, i]))>self.s_thre,1,0)
                       syn = sum(sum(self.dataset[j, k*self.c_space_num:(k+1)*self.c_space_num, l*self.c_space_num:(l+1)*self.c_space_num, i]))
                       V_spike, V = neuron1.neuron_ST(init[k,l,i] ,syn, self.s_thre,0,0)
                       self.spaceneuron[j,k,l,i] = V_spike 
                       init[k,l,i] = V
                       self.V_space[j,k,l,i] = V                                                                  
        return self.spaceneuron
    
    def Temporalprocessing(self):
        neuron1 = Neuron(self.if_st_neuron_clear)  
        init= np.full((np.shape(self.dataset)[1], int(np.shape(self.dataset)[2]/self.c_space_num), np.shape(self.dataset)[3]),0)
        self.V_time= np.full((np.shape(self.dataset)[0], int(np.shape(self.dataset)[1]/self.c_space_num), int(np.shape(self.dataset)[2]/self.c_space_num), np.shape(self.dataset)[3]),0)
        #self.timeneuron = np.full((int(np.shape(self.spaceneuron)[0]/self.t_window), np.shape(self.spaceneuron)[1], np.shape(self.spaceneuron)[2], np.shape(self.spaceneuron)[3]),0)
        self.timeneuron = np.full((int(np.shape(self.spaceneuron)[0]), np.shape(self.spaceneuron)[1], np.shape(self.spaceneuron)[2], np.shape(self.spaceneuron)[3]),0)
        for i in range(0, np.shape(self.timeneuron)[3]): ## event
            for j in range(0, np.shape(self.timeneuron)[0]):
                for k in range(0, np.shape(self.spaceneuron)[1]):
                    for l in range(0, np.shape(self.spaceneuron)[2]):
                         #syn = sum(self.spaceneuron[j*self.t_window:(j+1)*self.t_window, k, l, i])
                         syn = sum(self.spaceneuron[j:j+self.t_window, k, l, i])
                         V_spike, V = neuron1.neuron_ST(init[k,l,i] ,syn, self.t_thre,0,0)
                         self.timeneuron[j,k,l,i] = V_spike 
                         init[k,l,i] = V
                         self.V_time[j,k,l,i] = V  
        return self.timeneuron  

    def Stprocessing(self):  
        #self.stcore = np.full((np.shape(self.spaceneuron)[0], np.shape(self.spaceneuron)[1], np.shape(self.spaceneuron)[2], np.shape(self.spaceneuron)[3]),0)
        #timeneuron = np.full((np.shape(self.spaceneuron)[0], np.shape(self.spaceneuron)[1], np.shape(self.spaceneuron)[2], np.shape(self.spaceneuron)[3]),0)
        #for i in range(0, np.shape(self.stcore)[3]): ## event        
        #    for j in range(0, np.shape(self.timeneuron)[0]):
        #        timeneuron[j*self.t_window :(j+1)*self.t_window,:,:,i]  = self.timeneuron[j,:,:,i]

        #for i in range(0, np.shape(self.stcore)[3]): ## event 
           # for j in range(0, np.shape(self.stcore)[0]):
                #self.stcore[j,:,:,i] = self.spaceneuron[j,:,:,i]*self.timeneuron[j,:,:,i]
        #self.stcore= self.spaceneuron * timeneuron 
        self.stcore = self.spaceneuron * self.timeneuron
        

    def stspike(self):
        event_num = np.shape(self.dataset)[3]
        self.ST_spike= np.full((np.shape(self.stcore)[1],np.shape(self.stcore)[2],np.shape(self.stcore)[3]),0) 
        for i in range(0,event_num): ## event number
                for j in range(0, np.shape(self.stcore)[1]):
                    for k in range (0,np.shape(self.stcore)[2]):
                        self.ST_spike[j,k,i] = sum(self.stcore[:,j,k,i])
        self.ST_spike_1d = np.reshape(self.ST_spike,(np.shape(self.stcore)[1]*np.shape(self.stcore)[2],event_num))
        # stspike = cfg.code_path + '/data/stspike'
        # np.save(stspike, self.ST_spike_1d) 

    def stthresholdpatt(self, num):
        threshold_index = np.full((num, np.shape(self.ST_spike_1d)[1]),0)
        threshold_value = np.full((num, np.shape(self.ST_spike_1d)[1]),0)
        for i in range(0, np.shape(self.ST_spike_1d)[1]):  ## event
                for j in range(0, num):    
                    threshold_index[j,i] = np.argsort(self.ST_spike_1d[:,i])[-j-1]
                    threshold_value[j,i] = self.ST_spike_1d[threshold_index[j,i],i]
        SIMO_thres = cfg.code_path + '/data/reference_threshold'
        np.save(SIMO_thres, threshold_value)            
        return threshold_index, threshold_value 

    def stfeature(self):
        c_range = 60
        st_num = np.shape(self.ST_spike_1d)[0]
        threshold_index = np.full((int(st_num/c_range), np.shape(self.ST_spike_1d)[1]),0)
        threshold_value = np.full((int(st_num/c_range), np.shape(self.ST_spike_1d)[1]),0)
        for i in range(0, np.shape(self.ST_spike_1d)[1]):  ## event
                            for j in range(0,int(st_num/c_range)): ##
                                threshold_index[j,i] = np.argsort(self.ST_spike_1d[c_range*j:c_range*(j+1),i])[-1] + c_range*j
                                threshold_value[j,i] = self.ST_spike_1d[threshold_index[j,i],i]
        return threshold_index, threshold_value                        