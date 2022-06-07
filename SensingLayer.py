import sys
import os

import numpy as np
from BaseLayer import BaseLayer
from logger import Logger


class SensingLayer(BaseLayer):
    def __init__(self, dataset, t_interval, thre, polarity):
        #super(SensingLayer, self).__init__(input_channel, input_shape, output_channel)
        self.t_interval = t_interval
        self.thre = thre
        self.dataset = dataset
        self.polarity = polarity
        
    def DiffSensing(self): 
        dataset = self.dataset
        self.sensing_output = np.full((dataset.videos[0].shape[0], dataset.videos[0].shape[1], dataset.videos[0].shape[2], dataset.event_num),0)
        self.sensing_diff = np.full(( dataset.videos[0].shape[1], dataset.videos[0].shape[2]),0)
        for i in range(0, dataset.event_num):   ## event number
            Event_register = dataset.videos[i]
            for j in range(0,dataset.videos[0].shape[0]-self.t_interval-1):  ## frame number 
             
             self.sensing_diff= Event_register[j+self.t_interval] - Event_register[j]
                
             if self.polarity == 0:
                self.sensing_output[j,:,:,i] = np.where(np.logical_and(abs(self.sensing_diff) > self.thre
                            , abs(self.sensing_diff) < 250), 100, 0)
             elif self.polarity == 1: 
                self.sensing_output[j,:,:,i] = np.where(abs(self.sensing_diff) < self.thre, 0,(np.where(self.sensing_diff>0, 1, -1)))                 
             else:
                break

