import sys
import os
sys.path.append('..')

class DatasetBase(object):
    def __init__(self, root):
        self.root = root
    
