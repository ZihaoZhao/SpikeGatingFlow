import numpy as np
from numpy.random import seed
from numpy.random import randn
import cfg
class Neuron(object):
    def __init__(self, if_st_neuron_clear=False):
        self.if_st_neuron_clear = if_st_neuron_clear       
        pass
        

    def neuron_space_expert(self, syn, thres): 
        V_internal =  syn
        if V_internal > thres:
            V_spike = 1
            V_internal = 0
        else:
            V_spike = 0
        return V_spike

    def neuron_temporal_expert(self,  syn, stim, thres):
        if syn > 0.5:
             syn = 0.5 

        V_internal =  syn - 0.5 + stim   # 0.5 is a user defined value

        if V_internal >= thres:
            V_spike = 1
            V_internal = 0
        else:
            V_spike = 0 

        if V_internal >= 2:           ## max voltage range
            V_internal = 0
        elif V_internal < 0:          ## min voltage range
            V_internal = 0 
        else:
            pass  

        return V_spike, V_internal           

    def neuron_ST(self, init, syn, thres, st_ions, st_leakage): 
        V_internal =  init + syn + st_ions + st_leakage
        if V_internal >= thres:
            V_spike = 1
            V_internal = 0
        else:
            V_spike = 0 
            if self.if_st_neuron_clear:
                V_internal = 0

        if V_internal > 2:       ## max voltage range
              V_internal = 2
        elif V_internal < -0.5:  ## min voltage range
              V_internal = -0.5 
        else:
            pass     
        return V_spike, V_internal      

    def neuron_SIMO(self, init, input_value, threshold, output_num,ions, leakage):
                self.input_value = input_value
                self.output_num = output_num
                syn = input_value
                V_spike_index = 0
                V_spike_internal = np.empty(self.output_num,int)
                V_decoder = np.full((1,self.output_num),0) 
                V_internal = syn + init + ions - leakage
                for i in range(0, self.output_num):
                    if V_internal >= threshold[i]:
                            V_spike_internal[i] = 1
                    else:
                            V_spike_internal[i] = 0

                if np.all(V_spike_internal == 0): 
                        V_spike = 0
                        V_spike_index = 0
                else:
                        V_decoder = np.transpose(V_spike_internal) * threshold
                        V_spike_index =np.argsort(V_decoder)[-1]
                        V_spike = 1

                return V_spike, V_spike_index 

    def neuron_computation(self, st_1d_spike,threshold_value,input_num,output_num):
            n1 = Neuron()
            self.neuron_output = np.full((output_num,1),0) 
                       
            for i in range(0, output_num):    # event
                decoder = np.full((output_num,1),0)
                for j in range(0, input_num): #ST core number
                        init = 0
                        input_value = st_1d_spike[j,i]
                        threshold = threshold_value[j,:]
                        V_spike, V_spike_index = n1.neuron_SIMO(init, input_value, threshold, output_num,0, 0)
                        if V_spike == 1:
                            decoder[V_spike_index] +=1
                        else:
                            pass
                self.neuron_output[i] = np.argsort(decoder[:,0])[-1]
            print(self.neuron_output)                

    def neuron_CANN(self, syn, init, stim, resting, thres):
        V_internal =  init + syn + stim + resting
        if V_internal >= thres:
            V_spike = 1
            V_internal = 0
        else:
            V_spike = 0 

        return V_spike, V_internal  

    def neuron_weight(self,neuro_num,type):
        weight = np.full((neuro_num,neuro_num),0, dtype=float)
        if type == 0:  #random
            mean = 0
            stdv = 0.1
            for i in range(0, neuro_num):
                seed(1)
                weight[i,:] =  np.random.normal(mean, stdv, neuro_num)
            return weight 
        elif type == 1:  #sequence
             np.fill_diagonal(weight , 1) 
             weight = np.roll(weight,1,axis = 0) 
        else:
            pass

        return weight   

    def neuron_location(self, init, syn1, thres):  
        V_internal =  init + syn1 
        if V_internal >= thres:
            V_spike = 1
            V_internal = 0
        else:
            V_spike = 0 

        return V_spike, V_internal  



