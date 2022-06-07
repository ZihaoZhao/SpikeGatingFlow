import sys
import os
sys.path.append('..')

import numpy as np
import queue

class BaseLayer(object):
    def __init__(self, input_channel, input_shape, output_channel):
        self.input_channel = input_channel     ## Neuron number
        self.input_shape = input_shape
        self.output_channel = output_channel

        self.t = 0
        self.sample_i = 0
        self.output_history = list() 
        
        self.len_t = 0 
        
    def next_t(self):
        self.t += 1

    def previous_t(self):
        self.t -= 1
        if self.t <= 0:
            self.t = 0
        
    def clear_t(self):
        self.t = 0

    def next_sample(self):
        self.sample_i += 1
        self.clear_t()
        self.clear_historoy()

    def previous_sample(self):
        self.sample_i -= 1
        if self.sample_i <= 0:
            self.sample_i = 0
        self.clear_t()
        self.clear_historoy()

    def set_sample(self, i):
        self.sample_i = i
        self.clear_t()
        self.clear_historoy()

    def clear_historoy(self):
        self.output_history.clear()


    def set_len_t(self, t):
        self.len_t = t