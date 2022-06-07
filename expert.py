import numpy as np
from Neuron import Neuron
from Visualization import Visualization
from math import pow
import operator
from statistics import mean
from scipy.signal import convolve


class SGF_expert(Neuron):
    def __init__(self, args):
        self.args = args
        super().__init__()           


    def expert_space(self, resolution_col, resolution_row, data, thres, offset, \
                        thres_bit=1, thres_step=0.1, thres_inc_factor=[-1,1], if_vote=False, if_imbalance=False):
        # resolution:  active location resolution (>= 2), equal to space neuron power of 2
        # data:        input data format:[row, coloum, event number]
        # thres:       ST core neuron threshold and space neuron threshold: [ST_core threshold, space_threshold]
        # offset:      active scope offset either from row or coloum: [offset_row, offset_coloum]

        row_dimension = int(np.shape(data)[0])-offset[0]
        col_dimension = int(np.shape(data)[1])-offset[1]
        space_neuron = np.full((resolution_row*resolution_col*thres_bit*thres_bit,np.shape(data)[2]),0)            
        row_unit = int(row_dimension/resolution_row)
        col_unit = int(col_dimension/resolution_col)
        weight = np.full((row_dimension, col_dimension, np.shape(space_neuron)[0]), 0)
                
        j = 0 # space neuron
        for k in range(0, resolution_row):           # row
            for l in range(0, resolution_col):       # col            
                weight[row_unit*k:row_unit*(k+1), col_unit*l:col_unit*(l+1),j]=1
                j = j+1

        # thres_step = 0.1
        for t1 in range(thres_bit):
            for t2 in range(thres_bit):
                # print(int(thres[0]*pow((1+thres_inc_factor[0]*thres_step),t)),\
                #     int(thres[1]*pow((1+thres_inc_factor[1]*thres_step),t)))
                n1 = Neuron()
                syn = np.full((resolution_row*resolution_col,np.shape(data)[2]),0) 
                for i in range (0, np.shape(data)[2]): # event
                    for j in range(0, resolution_row*resolution_col):    # space neuron
                        if if_imbalance:
                            syn[j,i] = sum(sum(np.where(data[:,:,i]>(int(thres[0][j]*pow((1+thres_step),t1))), 1, 0) * weight[:,:,j]))  # thres0调节每个点的阈值
                            space_neuron[thres_bit*thres_bit*j+t1*thres_bit+t2,i]= n1.neuron_space_expert(syn[j,i],(int(thres[1][j]*pow((1+thres_step),t2))))                # thres1调节范围
                        else:
                            syn[j,i] = sum(sum(np.where(data[:,:,i]>(int(thres[0]*pow((1+thres_step),t1))), 1, 0) * weight[:,:,j]))  # thres0调节每个点的阈值
                            space_neuron[thres_bit*thres_bit*j+t1*thres_bit+t2,i]= n1.neuron_space_expert(syn[j,i],(int(thres[1]*pow((1+thres_step),t2))))                # thres1调节范围
        if if_vote:
            space_neuron_vote = np.full((resolution_row*resolution_col, np.shape(data)[2]), 0)  
            for i in range (0, np.shape(data)[2]): # event
                for j in range(0, resolution_row*resolution_col):    # space neuron
                    # print(sum(space_neuron[thres_bit*thres_bit*j:thres_bit*thres_bit*(j+1),i]))
                    if sum(space_neuron[thres_bit*thres_bit*j:thres_bit*thres_bit*(j+1),i]) >= thres_bit*thres_bit/2:
                        space_neuron_vote[j,i] = 1
                    else:
                        space_neuron_vote[j,i] = 0
            return space_neuron_vote  
        else:
            return space_neuron  

    def attention_expert(self, data, threshold):
        # data: data:        input data format:[row, coloum, event number]
        result_row_max = [0 for i in range (np.shape(data)[0])]
        result_row_min = [0 for i in range (np.shape(data)[0])]
        result_col_max = [0 for i in range (np.shape(data)[1])]
        result_col_min = [0 for i in range (np.shape(data)[1])]
        neuron_valid = np.full((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2]),0)
        result = np.full((4,np.shape(data)[2]),0)
        for i in range (0, np.shape(data)[2]): # event
                for j in range(0,np.shape(data)[0]):    # Row
                    for k in range(0, np.shape(data)[1]):
                        neuron_valid[j,k,i] = np.where(data[j,k,i]>threshold,1,0)  ## the absolute value 5 is user defined.
            
        for i in range (0, np.shape(data)[2]): # event
                for j in range(0,np.shape(data)[0]):    # Row
                    try:
                        result_row_max[j] = np.amax(np.where(neuron_valid[j,:,i] == 1))
                        result_row_min[j] = np.amin(np.where(neuron_valid[j,:,i] == 1))
                    except ValueError: 
                        pass 
                try:
                    result[0,i] =  np.amax(result_row_max)
                    result[1,i] =  np.min(np.nonzero(result_row_min))
                except ValueError: 
                    pass           
                for k in range(0, np.shape(data)[1]): 
                    try:                                           
                        result_col_max[k] = np.amax(np.where(neuron_valid[:,k,i] == 1))
                        result_col_min[k] = np.amin(np.where(neuron_valid[:,k,i] == 1)) 
                    except ValueError: 
                        pass
                try:  
                    result[2,i] =  np.amax(result_col_max)
                    result[3,i] =  np.min(np.nonzero(result_col_min))
                except ValueError: 
                    pass

        return result

    def expert_temporal(self, resolution, data, start, end, type, scale):
        # resolution:  temporal resolution
        # data:        input data format:[frame, row, coloum, event number]
        # start:       offset of the start points format: [row, coloum]
        # end:         offset of the end points format: [row, coloum]  
        # type:        temporal sequences: 0-(top, down), 1-(left, right)
        n1 = Neuron()
        row_length = end[0] - start[0]                                          # target zone row length
        col_length = end[1] - start[1]                                       # target zone coloum length
        temporal_neuron = np.full((int(pow(resolution,2)),np.shape(data)[3]),0)   # temporal neuron number
        row_unit = int(row_length/resolution)                                     # target zone row computational unit length 
        col_unit = int(col_length/resolution)                                     # target zone col computational unit length 
        #weight = np.full((row_length, col_length, row_length*col_length),0)       # target zone neuron weight information  
        weight = np.full((np.shape(data)[1], np.shape(data)[2], np.shape(data)[1]*np.shape(data)[2]),0) 
        modular_index = 0  
        index = 0
        location = [0 for i in range (4)]                        
        if type == 0:  ## top -> down  (left) + bottom - up (right)
           for k in range(0,resolution):            # row
               for l in range(0,resolution):        # col 
                    location[0]  =   row_unit*k 
                    location[1]  =   row_unit*(k +1) 
                    location[2]   =  col_unit*l + start[1]
                    location[3]   =  col_unit*(l+1) + start[1] 
                    if modular_index == 0 or  modular_index == 2:                   
                        for i in range(location[0],location[1]):                  # row number
                            for j in range(location[2],location[3]):              # coloum number
                                x_top = i-scale                                   # connection range
                                if x_top < location[0]: 
                                    x_top = location[0]
                                weight[x_top:i, j, index]  = 1 
                                weight[i,j,index] = 0
                                index = index +1
                    else:
                        for i in range(location[0],location[1]):    #row
                            for j in range(location[2],location[3]): #coloum
                                x_bottom = i+scale
                                if x_bottom > location[1]: 
                                    x_bottom = location[1]
                                weight[i:x_bottom, j, index]  = 1 
                                weight[i,j,index] = 0 
                                index = index +1                      
                    modular_index = modular_index +1
                    
        elif type == 1:  ## bottom->up (left) + top -> down )right
           for k in range(0,resolution):           # row
               for l in range(0,resolution):       # col 
                    location[0]  =   row_unit*k 
                    location[1]  =   row_unit*(k +1)  
                    location[2]   =  col_unit*l  + start[1]
                    location[3]   =  col_unit*(l +1) + start[1]
                    if modular_index == 1 or  modular_index == 3:                   
                        for i in range(location[0],location[1]):                  # row
                            for j in range(location[2],location[3]):              # coloum
                                x_top = i-scale                                   # connection range
                                if x_top < location[0]: 
                                    x_top = location[0]
                                weight[x_top:i, j, index]  = 1 
                                weight[i,j,index] = 0
                                index = index +1                      
                    else:
                        for i in range(location[0],location[1]):    #row
                            for j in range(location[2],location[3]): #coloum
                                x_bottom = i+scale
                                if x_bottom > location[1]: 
                                    x_bottom = location[1]         
                                weight[i:x_bottom, j, index]  = 1 
                                weight[i,j,index] = 0 
                                index = index +1                      
                    modular_index = modular_index +1
                    
        elif type == 2:  # left->right(left) + right->left (right)
           for k in range(0,resolution):           # row
               for l in range(0,resolution):       # col 
                    location[0]  =   row_unit*k 
                    location[1]  =   row_unit*(k +1)  
                    location[2]   =  col_unit*l  + start[1]
                    location[3]   =  col_unit*(l +1) + start[1]
                    if modular_index == 1 or  modular_index == 3:                   
                        for i in range(location[0],location[1]):                  # row
                            for j in range(location[2],location[3]):              # coloum
                                y_top = j-scale                                   # connection range
                                if y_top < location[2]: 
                                    y_top = location[2]
                                weight[i, y_top:j, index]  = 1 
                                weight[i,j,index] = 0
                                index = index +1                      

                    else:
                        for i in range(location[0],location[1]):    #row
                            for j in range(location[2],location[3]): #coloum
                                y_bottom = j+scale
                                if y_bottom > location[3]: 
                                    y_bottom = location[3]         
                                weight[i, j:y_bottom, index]  = 1 
                                weight[i,j,index] = 0 
                                index = index +1                      
                    modular_index = modular_index +1            

        else:
            pass                            
        spike_info =  np.full((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2],np.shape(data)[3]),0)
        temporal_index = np.full((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2],np.shape(data)[3]),0)
        temporal_neuron = np.full((resolution,resolution,np.shape(data)[3]),0)
        temporal_neuron_spike = np.full((resolution,resolution,np.shape(data)[3]),0)
        temporal_neuron_final = [0 for i in range (np.shape(data)[3])]
        
        for i in range(0, np.shape(data)[3]):           # event
              for j in range(1, np.shape(data)[0]):     # frame
                    index = 0
                    spike_info[0,:,:,i] = data[0,:,:,i]  # given the initial neuron states
                    for k in range(0,resolution):            # row
                        for l in range(0,resolution):        # col 
                            location[0]   =   row_unit*k 
                            location[1]   =   row_unit*(k +1)  
                            location[2]   =   col_unit*l + start[1]
                            location[3]   =   col_unit*(l+1) + start[1]
                            for m in range(location[0],location[1]):                  # row-element
                                for n in range(location[2],location[3]):              # coloum-element                   
                                    stim = data[j,m,n,i]
                                    syn = sum(sum(weight[:,:,index]*spike_info[j-1,:,:,i]))
                                    V_spike, V_internal = n1.neuron_temporal_expert(syn, stim, 0.5)                             
                                    if syn > 0 and stim == 1 :  # Hebbian learning
                                        temporal_index[j,m,n,i] = 1
                                    else:
                                        temporal_index[j,m,n,i] = 0
                                    spike_info[j,m,n,i] = V_spike 
                                    index = index +1
                    #v2 = Visualization(data)
                    #v2.generate_temporal_profiling( data[1,:,:,0], spike_info[0,:,:,0], temporal_index[1,:,:,0])
                    #v2.generate_weight_map(weight, location[0]*row_length +location[2], location[0]*row_length +location[3])
        for i in range(0, np.shape(data)[3]):            # event
              for k in range(0,resolution):              # row 
                  for l in range(0, resolution):                      
                        #temporal_neuron[k,l,i] = int(np.where(sum(sum(sum(temporal_index[:,row_unit*k:row_unit*(k+1), col_unit*l:col_unit*(l+1),i])))>thres,1,0)) 
                        temporal_neuron_spike[k,l,i] =   sum(sum(sum(temporal_index[:,row_unit*k:row_unit*(k+1), col_unit*l:col_unit*(l+1),i])))   
              temporal_neuron_final[i] = sum(sum(temporal_neuron_spike[:,:,i]))        
        return temporal_neuron_final


    def coord_transformation(self, element_location, col_length):  ## transfer modular local coord to global coord
        # location:            modular location, format[ row, coloum]
        # modular_size:        modular size, format:[row unit length, col unit length]
        # modular_index:       modular index 


        #new_location = modular_index*(modular_size[0]*modular_size[1]) + element_location[0]*modular_size[1] + element_location[1]
        new_location = element_location[0]*col_length + element_location[1]
        return new_location

        
    # def expert_topology(self,stcore,sense_scope):
    #     n1 = Neuron()
    #     frame = np.shape(stcore)[0]
    #     event_num = np.shape(stcore)[3]
    #     resolution = 6
    #     space_neuron_number = int(pow(resolution,2))
    #     stim = np.full((frame,int(pow(resolution,2)),event_num),0)
    #     for i in range (0,frame):
    #         space_neuron = self.expert_space(resolution, stcore[i,:,:,:], [0,10], [0,0])   ## find the most active location as the stimulus
    #         stim[i,:,:] = space_neuron                                                     ## generate active patterns

    #     weight = np.full((space_neuron_number,space_neuron_number,event_num),0)            ## define the weight matrix
    #     for i in range(0,event_num):  
    #         for j in range(1,frame):                                                       ## the weight has to be generated by frame            
    #             for  k in range(0,space_neuron_number):                                    
    #                 if stim[j,k,i] == 1:
    #                     ### This require further optimizations
    #                     for l in range (-sense_scope[1],sense_scope[1]):     ## sense range-coloum
    #                         for m in range (-sense_scope[0],sense_scope[0]):
    #                             if stim[j-1,k+l*resolution+m,i] == 1:
    #                                 weight[k,k+l*resolution+m,i] = 1                                      ## only generate weight when both frame has the connections.

    #     v1 = Visualization(stcore)
    #     v1.generate_weight_map(weight, 0, 9)
    #     spike_info = np.full((frame,space_neuron_number,event_num),0)                      
    #     temporal_index = np.full((frame,space_neuron_number,event_num),0) 
    #     topology_index = np.full((frame,event_num),0)  
    #     for i in range(0,event_num): 
    #         spike_info[0,:,i] = stim[0,:,i]   
    #         for j in range(1,frame):
    #             for k in range(0,space_neuron_number):
    #                 syn = sum(weight[k,:,5]*spike_info[j-1,:,i])                      # background computing
    #                 V_spike, V_internal = n1.neuron_temporal_expert(syn, stim[j,k,i], 0.5)                             
    #                 if syn > 0 and stim[j,k,i] == 1 :  # Hebbian learning
    #                     temporal_index[j,k,i] = 1
    #                 else:
    #                     temporal_index[j,k,i] = 0
    #                 spike_info[j,k,i] = V_spike 
    #             if sum(temporal_index[j,:,i])  > sum(stim[j,:,i]) * 0.8:
    #                 topology_index[j,i] = 1
    #     topology_neuron = [0 for i in range (event_num)]
    #     topology_spike = [0 for i in range (event_num)]        
    #     ref1 = sum(sum(temporal_index[:,:,0]))-6
    #     for i in range(0, event_num):
    #         topology_spike[i] = sum(sum(temporal_index[:,:,i]))
    #         topology_neuron[i] = np.where(topology_spike[i]>ref1,1,0)
    #     print(topology_neuron)
    #     print(topology_spike) 
    #     print(topology_index)           
    #     return stim,weight,temporal_index


    def expert_knn_train(self, train_data, train_id,resolution,index, id_start, id_end):

        pattern_num = 0
        res = np.power(resolution,2)
        event0_code = np.full((np.shape(train_id)[0],np.shape(train_data)[1]),0)
        event_knowledge_clock = np.full((resolution, resolution),0)
        event_knowledge_anticlock = np.full((resolution, resolution),0)        
        index0 = 0
        for i in range (0, np.shape(train_id)[0]):
            if train_id[i] == index:
                event0_code[index0] = train_data[i,:]
                index0 = index0 + 1
                pattern_num = pattern_num + 1
        event0_code = event0_code[~np.all(event0_code == 0, axis=1)]        
        xdata = np.full((np.shape(event0_code)[0],res),0)
        ydata = np.full((np.shape(event0_code)[0],res),0)
        for j in range (0, np.shape(event0_code)[0]):
            zdata = event0_code[j,8:10]    ## temporal domain
            xydata = event0_code[j,id_start:id_end]
            if zdata[0] == 1 and  zdata[1] == 0:
                for k in range (0, res):
                    if xydata[k] == 1:
                        xdata[j,k] = k//resolution
                        ydata[j,k] = k%resolution
                        event_knowledge_clock[xdata[j,k],ydata[j,k]] = 1    
            #v1 =  Visualization(event0_code) 
            #v1.generate_spike_code(xdata,ydata,1,index) 

            if zdata[0] == 0 and  zdata[1] == 1:
                for k in range (0, res):
                    if xydata[k] == 1:
                        xdata[j,k] = k//resolution
                        ydata[j,k] = k%resolution
                        event_knowledge_anticlock[xdata[j,k],ydata[j,k]] = 1               

            #v1 =  Visualization(event0_code) 
            #v1.generate_spike_code(xdata,ydata,0,index) 
        # print(pattern_num)     
        return event_knowledge_clock, event_knowledge_anticlock



    def expert_knn_test(self,test_data, resolution, event_knowledge,event_knowledge_anticlock,id_start, id_end):                       
        res = np.power(resolution,2)
        event0_code = test_data
        test_code= np.full((resolution, resolution),0)
        xdata = [0  for x in range(np.power(resolution,2))]
        ydata = [0  for x in range(np.power(resolution,2))]

        for j in range (0, np.shape(event0_code)[0]):
            zdata = event0_code[8:10]    ## temporal domain
            xydata = event0_code[id_start:id_end]
            for k in range (0, res):
                if xydata[k] == 1:
                    xdata[k] = k//resolution
                    ydata[k] = k%resolution
                    test_code[xdata[k],ydata[k]] = 1

        score = np.full((resolution,resolution,10),0)
        score_p = [0  for x in range(10)]
        score_n = [0  for x in range(10)]  
        result = [0  for x in range(10)]  

        if zdata[0] == 1 and  zdata[1] == 0:  ## clockwise
            for i in range(0,np.shape(event_knowledge)[2]):
                for j in range (0, resolution):
                    for k in range(0,resolution):
                        if test_code[j,k] == event_knowledge[j,k,i] and test_code[j,k] == 1:
                            score[j,k,i] = 1
                            score_p[i] = score_p[i]+1
                        elif test_code[j,k] != event_knowledge[j,k,i] and test_code[j,k] == 1:
                            score[j,k,i] = -1
                            score_n[i] = score_n[i]-1
                        else:
                            pass 
                result[i] =  score_p[i] + score_n[i] 
                #print(score_p,score_n)    

        if zdata[0] == 0 and  zdata[1] == 1:  ## clockwise
            for i in range(0,np.shape(event_knowledge_anticlock)[2]):
                for j in range (0, resolution):
                    for k in range(0,resolution):
                        if test_code[j,k] == event_knowledge_anticlock[j,k,i] and test_code[j,k] == 1:
                            score[j,k,i] = 1
                            score_p[i] = score_p[i]+1
                        elif test_code[j,k] != event_knowledge_anticlock[j,k,i] and test_code[j,k] == 1:
                            score[j,k,i] = -1
                            score_n[i] = score_n[i]-1
                        else:
                            pass 
                result[i] =  score_p[i] + score_n[i] 
                #print(score_p,score_n)   

        #print(score_total)

        #print(score_p)
        #print(score_n)
        #result_p = np.argmax(score_p)
        #result_n = np.argmin(score_n)   
        #result = [result_p, result_n]  
        #result = np.argmax(result)  
        result_return = np.where(result ==np.amax(result))
        return result_return

    def expert_overlap(self,data, type):
        # data:        input data format:[ frame,row, coloum, event number]
        # type:        1: to check if there any active areas are overlapped.
        frame = np.shape(data)[0]
        event_num = np.shape(data)[3]



    def expert_temporal_poc(self, resolution, start, end, type, scale, data):
        n1 = Neuron()
        row_length = end[0] - start[0]                                          # target zone row length
        col_length = end[1] - start[1]                                       # target zone coloum length
        #temporal_neuron = np.full((int(pow(resolution,2)),np.shape(data)[3]),0)   # temporal neuron number
        row_unit = int(row_length/resolution)                                     # target zone row computational unit length 
        col_unit = int(col_length/resolution)                                     # target zone col computational unit length 
        weight = np.full((row_length, col_length, row_length*col_length),0)       # target zone neuron weight information  
        #weight = np.full((np.shape(data)[1], np.shape(data)[2], np.shape(data)[1]*np.shape(data)[2]),0) 
        modular_index = 0  
        index = 0
        location = [0 for i in range (4)]                        
        if type == 0:  ## pre: left
            for k in range(0,resolution):              # define the boundary of the area
                for l in range(0,resolution):         
                    location[0]  =  row_unit*k                 # row start address
                    location[1]  =  row_unit*(k +1)            # row end address 
                    location[2]  =  col_unit*l + start[1]      # coloum start address 
                    location[3]  =  col_unit*(l+1) + start[1]  # coloum end address               
                    print(location) 
                    if modular_index == 0:                
                        for i in range(location[0],location[1]):                  # row
                            for j in range(location[2],location[3]):              # coloum
                                y_top = j-scale                                   # connection range
                                if y_top < location[2]: 
                                    y_top = location[2]
                                weight[i, y_top:j, index]  = 1 
                                weight[i,j,index] = 0
                                index = index +1                                  # neuron index 
                    modular_index = modular_index +1                                                                         
        
        elif type == 1: ## pre: right
            for k in range(0,resolution):              # define the boundary of the area
                for l in range(0,resolution):         
                    location[0]  =  row_unit*k                 # row start address
                    location[1]  =  row_unit*(k +1)            # row end address 
                    location[2]  =  col_unit*l + start[1]      # coloum start address 
                    location[3]  =  col_unit*(l+1) + start[1]  # coloum end address               
                    print(location) 
                    if modular_index == 0:                
                        for i in range(location[0],location[1]):                  # row
                            for j in range(location[2],location[3]):              # coloum
                                y_top = j+scale                                   # connection range
                                if y_top > location[3]: 
                                    y_top = location[3]
                                weight[i, j:y_top, index]  = 1 
                                weight[i,j,index] = 0
                                index = index +1                                  # neuron index 
                    modular_index = modular_index +1                            
        else:
            pass
        v1 = Visualization(weight)   
        v1.generate_weight_map(weight,21,25)   


        ## start neuroal computation        
        spike_info =  np.full((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2],np.shape(data)[3]),0)
        temporal_index = np.full((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2],np.shape(data)[3]),0)
        temporal_neuron = np.full((resolution,resolution,np.shape(data)[3]),0)
        temporal_neuron_spike = np.full((resolution,resolution,np.shape(data)[3]),0)
        temporal_neuron_final = [0 for i in range (np.shape(data)[3])]
        
        for i in range(0, np.shape(data)[3]):           # event
              for j in range(1, np.shape(data)[0]):     # frame
                    index = 0
                    spike_info[0,:,:,i] = data[0,:,:,i]  # given the initial neuron states
                    for k in range(0,resolution):            # row
                        for l in range(0,resolution):        # col 
                            location[0]   =   row_unit*k 
                            location[1]   =   row_unit*(k +1)  
                            location[2]   =   col_unit*l + start[1]
                            location[3]   =   col_unit*(l+1) + start[1]
                            for m in range(location[0],location[1]):                  # row-element
                                for n in range(location[2],location[3]):              # coloum-element                   
                                    stim = data[j,m,n,i]
                                    syn = sum(sum(weight[:,:,index]*spike_info[j-1,:,:,i]))
                                    V_spike, V_internal = n1.neuron_temporal_expert(syn, stim, 0.5)                             
                                    if syn > 0 and stim == 1 :  # Hebbian learning
                                        temporal_index[j,m,n,i] = 1
                                    else:
                                        temporal_index[j,m,n,i] = 0
                                    spike_info[j,m,n,i] = V_spike 
                                    index = index +1
                    #v2 = Visualization(data)
                    #v2.generate_temporal_profiling( data[1,:,:,0], spike_info[0,:,:,0], temporal_index[1,:,:,0])
                    #v2.generate_weight_map(weight, location[0]*row_length +location[2], location[0]*row_length +location[3])
        for i in range(0, np.shape(data)[3]):            # event
              for k in range(0,resolution):              # row 
                  for l in range(0, resolution):                      
                        #temporal_neuron[k,l,i] = int(np.where(sum(sum(sum(temporal_index[:,row_unit*k:row_unit*(k+1), col_unit*l:col_unit*(l+1),i])))>thres,1,0)) 
                        temporal_neuron_spike[k,l,i] =   sum(sum(sum(temporal_index[:,row_unit*k:row_unit*(k+1), col_unit*l:col_unit*(l+1),i])))   
              temporal_neuron_final[i] = sum(sum(temporal_neuron_spike[:,:,i])) 
        print(temporal_neuron_spike)       
        return temporal_neuron_final,temporal_neuron_spike    

    def expert_space_event10(self, space_neuron,resolution):
        event_num = np.shape(space_neuron)[1]
        resp = [0 for i in range (event_num)]
        index = [0 for i in range (event_num)]

        for i in range(0, event_num):
            test_data = space_neuron[:,i].reshape(resolution, resolution)
            if np.any(test_data[:,5] == 1):
                resp[i] = 1


        return resp    

    def expert_hopfield(self, data, type,factor):    
        # data:        input data format:[frame, row, coloum, event number]
        # type:        detect temporal movements: 0-(top, down), 1-(bottom, up) , 2-(left,right), 3(right, left) 
        # factor:      the threshold scale factor  (default = 0.7)

        frame_num = np.shape(data)[0]
        #frame_num = 10
        row_num = np.shape(data)[1]
        col_num = np.shape(data)[2]       
        target = np.full((frame_num,row_num,col_num,2),0)
        target_spike = np.full((frame_num,row_num,col_num,2),0)
        target_all = np.full((frame_num,1),0)

        if type == 0:
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if j < active_pixels[0][l]:     ##detect the movement of row
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])   
            #print(target)               
        elif type == 1: 
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if j > active_pixels[0][l]:     ##detect the movement of row
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])   
        elif type == 2:
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if k > active_pixels[1][l]:     ##detect the movement of col
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])
        elif type == 3:
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if k < active_pixels[1][l]:     ##detect the movement of col
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])    
        else:
            pass  
            ### filtering sparse events that caused by the devices

        res = 5
        distrubtion_resolution = list(i for i in range (0, row_num,res))
        target_spike_hist = np.full((frame_num,len(distrubtion_resolution)-1),0)    
        if type == 0  or type == 1:
            #threshold = 40  ## if below 40, there are background noises
            for i in range(1,frame_num):
                active_pixels = np.nonzero(target_spike[i,:,:,0])
                target_spike_hist[i,:] = np.histogram(active_pixels[0], bins=distrubtion_resolution)[0]

        elif type == 2 or type == 3:
            #threshold = 30  ## if below 40, there are background noises
            for i in range(1,frame_num):            
                active_pixels = np.nonzero(target_spike[i,:,:,0])
                target_spike_hist[i,:] = np.histogram(active_pixels[1], bins=distrubtion_resolution)[0] 

        #position_spike = sum(target_spike_hist[:,0:-1])  


        ### check the movement among timing
        threshold = 15 ## #25
        target_spike_hist [target_spike_hist < threshold] = 0   

        ## find active area index
        target_areas_index = np.full((frame_num,len(distrubtion_resolution)-1),-1)   
        for i in range(1,frame_num):  ## experts sequence
            for j in range(0, len(distrubtion_resolution)-1): ## active area index
                if target_spike_hist[i,j] > 0:
                    target_areas_index[i,j] =j
                else:
                    target_areas_index[i,j] =-1   ## there is no active areas                       

        
        ## detect movement directions
        areas_index = []
        for i in range(0, frame_num):
            temporal = max(target_areas_index[i,:])
            if temporal == -1:
                areas_index.append(-1)
                #pass
            else:
                areas_index.append(max(target_areas_index[i,:]))

                        
        # print(areas_index)
        location_bit = [0 for i in range(0,4)]
        ## start to decode areas_index(neural activities)
        active_index_1 = [i for i in areas_index if i >= 0]   ## delete non active area -1 
        time_bit = 100 
        number = len(set(active_index_1))
        spike_flag = 0
        if number != 0:    ## if there is an activity or activities
            if type == 0 or type == 1:
                ##calculat the decoding values:
                max_value = max(active_index_1)  
                min_value = min(active_index_1)
                decoding_value = max_value - min_value
                ##check the spike intensities distributions
                position_flag = 0
                frame_index = [i for i, j in enumerate(areas_index) if j == max_value]          ## find the max_position frame index. This can be either max_position or min_position                
                while position_flag == 0:
 
                    active_pixels = np.nonzero(target[frame_index[0]-1 ,:,:,0])                      ## check the individual neuron intensitiy

                    spike_intensity_distribution = np.histogram(active_pixels[0], bins=distrubtion_resolution)[0]  ## check the spike intensity disutrbutions
                    required_position = [i for i, j in enumerate(spike_intensity_distribution) if j > 5]  ##10        ## handcraft number 10, is the number less than 10, this indicates noises.
                    if len(required_position) > 1:
                        position_flag = 1
                    else:
                        frame_index[0] = frame_index[0] +1

                    if frame_index[0] >= np.shape(target)[0]:
                        position_flag = 1
                        spike_flag = 1


                if spike_flag == 0:
                    required_position_mean_spike = np.zeros_like(required_position)
                    for i in range(0, len(required_position)):
                        target_area = target[frame_index[0]-1,required_position[i]*res:(required_position[i]+1)*res,:,0]
                        required_position_mean_spike[i] = np.true_divide(target_area.sum(),(target_area!=0).sum())

                    if required_position_mean_spike[0] > required_position_mean_spike[-1]:         ## if the bottom area spike intensities over the top spike intensities
                        spike_flow = 'top->down'                                                 #the spike flow is defined as the spike intensities flow from small values to the big values.
                    elif required_position_mean_spike[0] < required_position_mean_spike[-1]:    # if the bottom area spike intensities smaller the top spike intensities    
                        spike_flow = 'bottom->up' 
                    else:
                        spike_flow = 'unknown'    
                else:
                    spike_flow = 'unknown'    
                # print('spike_flow:',spike_flow) 

            elif type == 2 or type == 3:   
                ##calculat the decoding values:
                max_value = max(active_index_1)  
                min_value = min(active_index_1)
                decoding_value = max_value - min_value
                ##check the spike intensities distributions
                position_flag = 0
                frame_index = [i for i, j in enumerate(areas_index) if j == max_value]          ## find the max_position frame index. This can be either max_position or min_position                
                while position_flag == 0:

                    #frame_index = 3
                    active_pixels = np.nonzero(target[frame_index[0]-1 ,:,:,0])                      ## check the individual neuron intensitiy

                    spike_intensity_distribution = np.histogram(active_pixels[1], bins=distrubtion_resolution)[0]  ## check the spike intensity disutrbutions
                    required_position = [i for i, j in enumerate(spike_intensity_distribution) if j > 5]          ## handcraft number 10, is the number less than 10, this indicates noises.
                    if len(required_position) > 1:
                        position_flag = 1
                    else:
                        frame_index[0] = frame_index[0] +1

                    if frame_index[0] >= np.shape(target)[0]:
                        position_flag = 1
                        spike_flag = 1


                if spike_flag == 0:
                    required_position_mean_spike = np.zeros_like(required_position)
                    for i in range(0, len(required_position)):
                        target_area = target[frame_index[0]-1,:,required_position[i]*res:(required_position[i]+1)*res,0]
                        required_position_mean_spike[i] = np.true_divide(target_area.sum(),(target_area!=0).sum())

                    if required_position_mean_spike[0] > required_position_mean_spike[-1]:         ## if the bottom area spike intensities over the top spike intensities
                        spike_flow = 'right->left'                                                 #the spike flow is defined as the spike intensities flow from small values to the big values.
                    elif required_position_mean_spike[0] < required_position_mean_spike[-1]:    # if the bottom area spike intensities smaller the top spike intensities    
                        spike_flow = 'left->right' 
                    else:
                        spike_flow = 'unknown'       
                else:
                    spike_flow = 'unknown'    
                # print('spike_flow:',spike_flow)   
            else:
                print('hopfield network type error')

            #print('test')
            max_position = [i for i, j in enumerate(areas_index) if j == max_value]
            #print(max_position)
            min_position = [i for i, j in enumerate(areas_index) if j == min_value] 
            #print(min_position)
            v1 = Visualization(target[:,:,:,0])  
            '''
            if decoding_value == number-1 and v1.areConsecutive(max_position,len(max_position))== True and decoding_value != 0 :  ## if there is a pattern format : 3-2-1 or 3-2
            ## the pattern format: 3-2-1 or 3-2
                max_position = [i for i, j in enumerate(areas_index) if j == max_value]
                if type == 0 or type == 1:
                    if max_position[0] < min_position[0]:  ## if the movement is from top-> down
                        direction_flow = 'top->down' 
                    elif  max_position[0] > min_position[0]:   
                        direction_flow = 'bottom->up' 
                    else:
                        direction_flow = 'unknwon'   
                elif type == 2 or type == 3:  
                    if max_position[0] < min_position[0]:  ## if the movement is from top-> down
                        direction_flow = 'right->left' 
                    elif  max_position[0] > min_position[0]:   
                        direction_flow = 'left->right' 
                    else:
                        direction_flow = 'unknwon'  
                print('direction_flow:',direction_flow)                
                if direction_flow == spike_flow:
                    hopfield_bit = 1   ## there is an required pattern activities

                    ## dectect lift right location
                    if type == 2 or type ==  3:
                        if mean(active_index_1) > 7: ## check whether the active areas are at left
                            location_bit[0] = 1

                        if mean(active_index_1) <= 7: ## check whether the active areas are at right
                            location_bit[1] = 1   

                    ## detect start timing
                    frame = [i for i, j in enumerate(areas_index) if j >=0]   
                    time_bit = frame[0]   

                else:
                    hopfield_bit = 0    
            '''    
            #if decoding_value == number-1 and v1.areConsecutive([abs(x) for x in max_position],len(max_position)) == False and decoding_value != 0:  ## if there is a pattern format : 3-2-3 or 2-3-2
            if 1 == 1:  ## if there is a pattern format : 3-2-3 or 2-3-2
            ## the pattern format: 3-2-3 or 1-2-1   there is a local minial  points 
            #    max_position = [i for i, j in enumerate(areas_index) if j == max_value]
                i = 1
                pit_point_flap = 0
                sense_flag = 0
                sense_range = 4
                j = 1

                if type == 0 or type == 1:
                    while pit_point_flap ==0: 
                        while sense_flag == 0:
                            try:
                                if areas_index[i] != -1 and areas_index[i+j] != -1 :
                                    if areas_index[i] > areas_index[i+j] and abs(areas_index[i] - areas_index[i+j])<3: #5
                                            direction_flow = 'top->down' 
                                            if direction_flow == spike_flow:
                                                pit_point_flap = 1
                                                hopfield_bit = 1
                                                time_bit = i  
                                                sense_flag = 1 
                                    elif areas_index[i] < areas_index[i+j] and abs(areas_index[i] - areas_index[i+j])<3: #5
                                            direction_flow = 'bottom->up'
                                            if direction_flow == spike_flow: 
                                                pit_point_flap = 1
                                                hopfield_bit = 1
                                                time_bit = i 
                                                sense_flag = 1                                                  
                                    else:
                                        pass
                            except:
                                return [0], 0, np.array([0,0,0,0]), 0

                            if j >= sense_range:
                                sense_flag = 1
                                j = 1
                            else:
                                j = j+1                                
                        sense_flag = 0
                        
                        if i == len(areas_index) -sense_range -1:  ## if match the last second
                            hopfield_bit = 0
                            pit_point_flap = 1                                    
                        if pit_point_flap == 0:
                            i = i+1
                        


                elif type == 2 or type == 3:  
                    while pit_point_flap ==0: ## loop in area index
                        while sense_flag == 0:  ## loop in the sense range
                            try:
                                if areas_index[i] != -1 and areas_index[i+j] != -1:
                                    if areas_index[i] > areas_index[i+j] and abs(areas_index[i] - areas_index[i+j])<=3:   
                                            direction_flow = 'right->left' 
                                            if direction_flow == spike_flow:
                                                pit_point_flap = 1
                                                hopfield_bit = 1
                                                time_bit = i 
                                                sense_flag = 1 
                                    elif areas_index[i] < areas_index[i+j] and abs(areas_index[i] - areas_index[i+j])<=3:
                    
                                            direction_flow = 'left->right' 
                                            if direction_flow == spike_flow:
                                                pit_point_flap = 1
                                                hopfield_bit = 1
                                                time_bit = i 
                                                sense_flag = 1  
                                    else:
                                        pass
                            except:
                                return [0], 0, np.array([0,0,0,0]), 0
                                
                            if j >= sense_range:
                                sense_flag = 1
                                j=1
                            else:
                                j = j+1
                        sense_flag = 0
                        
                        if i == len(areas_index) - sense_range-1:  ## if match the last second
                            hopfield_bit = 0
                            pit_point_flap = 1                                   
                        if pit_point_flap == 0:
                            i = i+1
                  

                if hopfield_bit == 1:  

                    if type == 2 or type ==  3:
                        if areas_index[time_bit] <19: ## left areas
                            location_bit[0] = 1

                        if areas_index[time_bit] >=19: ## right areas  
                            location_bit[1] = 1  

                    if type == 0 :
                        max_value = max(active_index_1)  
                        min_value = min(active_index_1)
                        if abs(max_value-min_value) >= 5:
                            if areas_index[time_bit]>22: ## top areas
                                location_bit[2] = 1
                            if areas_index[time_bit]<=22: ## bottom areas
                                location_bit[3] = 1 
                        else:
                            if areas_index[time_bit]>19: ## top areas
                                location_bit[2] = 1
                            if areas_index[time_bit]<=19: ## bottom areas
                                location_bit[3] = 1                                                                
                    if type == 1:
                        if areas_index[time_bit]<=17: ## bottom areas  
                            location_bit[3] = 1   
                        if areas_index[time_bit]>17: ## top areas  
                            location_bit[2] = 1                                                                     
                    # print('direction_flow:',direction_flow) 
            #elif decoding_value != number-1:
            #    hopfield_bit = 0 

            #elif decoding_value == 0:    ## if there is only one activities
            #    hopfield_bit = 0 

            else:
                hopfield_bit = -1
        else:
            hopfield_bit = 0
            # print('no active patterns')           

        # print(hopfield_bit)
        return areas_index, hopfield_bit,location_bit, time_bit 


    def expert_hopfield_v1(self, data, type,factor):    
        # data:        input data format:[frame, row, coloum, event number]
        # type:        detect temporal movements: 0-(top, down), 1-(bottom, up) , 2-(left,right), 3(right, left) 
        # factor:      the threshold scale factor  (default = 0.7)

        frame_num = 3
        row_num = np.shape(data)[1]
        col_num = np.shape(data)[2]       
        target = np.full((frame_num,row_num,col_num,2),0)
        target_spike = np.full((frame_num,row_num,col_num,2),0)
        target_all = np.full((frame_num,1),0)

        if type == 0:
            for i in range(1, frame_num):
                previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if j < active_pixels[0][l]:     ##detect the movement of row
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])   
            #print(target)               
        elif type == 1: 
            for i in range(1, frame_num):
                previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if j > active_pixels[0][l]:     ##detect the movement of row
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])   
        elif type == 2:
            for i in range(1, frame_num):
                previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if k > active_pixels[1][l]:     ##detect the movement of col
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])
        elif type == 3:
            for i in range(1, frame_num):
                previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if k < active_pixels[1][l]:     ##detect the movement of col
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])                                            

        return target,target_all    

    def expert_hopfield2(self, data, type,factor,hist_threshold):    
        # data:        input data format:[frame, row, coloum, event number]
        # type:        detect temporal movements: 0-(top, down), 1-(bottom, up) 
        # factor:      the threshold scale factor  (default = 0.7)
        # hist_threshold:  spike histgram threshold

        frame_num = np.shape(data)[0]
        row_num = np.shape(data)[1]
        col_num = np.shape(data)[2]       
        target = np.full((frame_num,row_num,col_num,2),0)
        target_spike = np.full((frame_num,row_num,col_num,2),0)
        target_all = np.full((frame_num,1),0)

        if type == 0:
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if j < active_pixels[0][l]:     ##detect the movement of row
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                # print("target pixels:", target_all[i])   
            #print(target)               
        elif type == 1: 
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if j > active_pixels[0][l]:     ##detect the movement of row
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])   
        elif type == 2:
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if k > active_pixels[1][l]:     ##detect the movement of col
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])
        elif type == 3:
            for i in range(1, frame_num):
                if i == 0:
                    previous_event = data[i,:,:] #self checking
                else:
                    previous_event = data[i-1,:,:]
                active_pixels = np.nonzero(previous_event)
                #print("activel pixels:", len(active_pixels[0]))
                for j in range(0,row_num):
                    for k in range(0, col_num):
                        if data[i,j,k] == 1:  # if there is an event  
                            for l in range(0, len(active_pixels[0])):
                                if k < active_pixels[1][l]:     ##detect the movement of col
                                    target[i,j,k,0] = target[i,j,k,0]+1
                        if target[i,j,k,0] >  len(active_pixels[0]) *factor:
                           target_spike [i,j,k,0] = 1

                        #target[i,j,k,0] = target[i,j,k,0]/len(active_pixels[0])
                target_all[i] = sum(sum(target_spike [i,:,:,0]))   
                #print("target pixels:", target_all[i])    
        else:
            pass   
            ### filtering sparse events that caused by the devices

        res = 5
        distrubtion_resolution = list(i for i in range (0, row_num,res))
        target_spike_hist = np.full((frame_num,len(distrubtion_resolution)-1),0)    
        if type == 0  or type == 1:
            #threshold = 40  ## if below 40, there are background noises
            for i in range(1,frame_num):
                active_pixels = np.nonzero(target_spike[i,:,:,0])
                target_spike_hist[i,:] = np.histogram(active_pixels[0], bins=distrubtion_resolution)[0]

        elif type == 2 or type == 3:
            #threshold = 30  ## if below 40, there are background noises
            for i in range(1,frame_num):            
                active_pixels = np.nonzero(target_spike[i,:,:,0])
                target_spike_hist[i,:] = np.histogram(active_pixels[1], bins=distrubtion_resolution)[0] 

        #position_spike = sum(target_spike_hist[:,0:-1])  


        ### check the movement among timing
        #hist_threshold = 15 ## #25
        target_spike_hist [target_spike_hist < hist_threshold] = 0   

        ## find active area index
        target_areas_index = np.full((frame_num,len(distrubtion_resolution)-1),-1)   
        for i in range(1,frame_num):  ## experts sequence
            for j in range(0, len(distrubtion_resolution)-1): ## active area index
                if target_spike_hist[i,j] > 0:
                    target_areas_index[i,j] =j
                else:
                    target_areas_index[i,j] =-1   ## there is no active areas                       

        
        ## detect movement directions
        areas_index = []
        for i in range(0, frame_num):
            temporal = max(target_areas_index[i,:])
            if temporal == -1:
                areas_index.append(-1)
                #pass
            else:
                areas_index.append(max(target_areas_index[i,:]))

                        
        # print(areas_index)
        location_bit = [0 for i in range(0,4)]
        ## start to decode areas_index(neural activities)
        active_index_1 = [i for i in areas_index if i >= 0]   ## delete non active area -1 
        time_bit = 100 
        number = len(set(active_index_1))
        spike_flag = 0
        feature_ascent = 0
        feature_swing = 0
        if number != 0:    ## if there is an activity or activities
            if type == 0 or type == 1:
                ##calculat the decoding values:
                max_value = max(active_index_1)  
                min_value = min(active_index_1)
                decoding_value = max_value - min_value
                ##check the spike intensities distributions
                position_flag = 0
                frame_index = [i for i, j in enumerate(areas_index) if j == max_value]          ## find the max_position frame index. This can be either max_position or min_position                
                while position_flag == 0:
 
                    try:
                        active_pixels = np.nonzero(target[frame_index[0]-1 ,:,:,0])                      ## check the individual neuron intensitiy
                    except:
                        return areas_index, 0, 0

                    spike_intensity_distribution = np.histogram(active_pixels[0], bins=distrubtion_resolution)[0]  ## check the spike intensity disutrbutions
                    required_position = [i for i, j in enumerate(spike_intensity_distribution) if j > 5]  ##10        ## handcraft number 10, is the number less than 10, this indicates noises.
                    if len(required_position) > 1:
                        position_flag = 1
                    else:
                        frame_index[0] = frame_index[0] +1
                    
                    if frame_index[0] >= np.shape(target)[0]:
                        position_flag = 1
                        spike_flag = 1

                if spike_flag == 0:
                    required_position_mean_spike = np.zeros_like(required_position)
                    for i in range(0, len(required_position)):
                        target_area = target[frame_index[0]-1,required_position[i]*res:(required_position[i]+1)*res,:,0]
                        required_position_mean_spike[i] = np.true_divide(target_area.sum(),(target_area!=0).sum())

                    if required_position_mean_spike[0] > required_position_mean_spike[-1]:         ## if the bottom area spike intensities over the top spike intensities
                        spike_flow = 'top->down'                                                 #the spike flow is defined as the spike intensities flow from small values to the big values.
                    elif required_position_mean_spike[0] < required_position_mean_spike[-1]:    # if the bottom area spike intensities smaller the top spike intensities    
                        spike_flow = 'bottom->up' 
                    else:
                        spike_flow = 'unknown' 
                else:
                    spike_flow = 'unknown'            
                # print('spike_flow:',spike_flow) 
            elif type == 2 or type == 3:   
                ##calculat the decoding values:
                max_value = max(active_index_1)  
                min_value = min(active_index_1)
                decoding_value = max_value - min_value
                ##check the spike intensities distributions
                position_flag = 0
                frame_index = [i for i, j in enumerate(areas_index) if j == max_value]          ## find the max_position frame index. This can be either max_position or min_position                
                while position_flag == 0:

                    #frame_index = 3
                    try:
                        active_pixels = np.nonzero(target[frame_index[0]-1 ,:,:,0])                      ## check the individual neuron intensitiy
                    except:
                        return areas_index, 0, 0
                        
                    spike_intensity_distribution = np.histogram(active_pixels[1], bins=distrubtion_resolution)[0]  ## check the spike intensity disutrbutions
                    required_position = [i for i, j in enumerate(spike_intensity_distribution) if j > 5]          ## handcraft number 10, is the number less than 10, this indicates noises.
                    if len(required_position) > 1:
                        position_flag = 1
                    else:
                        frame_index[0] = frame_index[0] +1

                required_position_mean_spike = np.zeros_like(required_position)
                for i in range(0, len(required_position)):
                    target_area = target[frame_index[0]-1,:,required_position[i]*res:(required_position[i]+1)*res,0]
                    required_position_mean_spike[i] = np.true_divide(target_area.sum(),(target_area!=0).sum())

                if required_position_mean_spike[0] > required_position_mean_spike[-1]:         ## if the bottom area spike intensities over the top spike intensities
                    spike_flow = 'right->left'                                                 #the spike flow is defined as the spike intensities flow from small values to the big values.
                elif required_position_mean_spike[0] < required_position_mean_spike[-1]:    # if the bottom area spike intensities smaller the top spike intensities    
                    spike_flow = 'left->right' 
                else:
                    spike_flow = 'unknown'    
                # print('spike_flow:',spike_flow)   
            else:
                print('hopfield network type error')

            #print('test')
            max_position = [i for i, j in enumerate(areas_index) if j == max_value]
            #print(max_position)
            min_position = [i for i, j in enumerate(areas_index) if j == min_value] 
            #print(min_position)
            v1 = Visualization(target[:,:,:,0])  
 
            #if decoding_value == number-1 and v1.areConsecutive([abs(x) for x in max_position],len(max_position)) == False and decoding_value != 0:  ## if there is a pattern format : 3-2-3 or 2-3-2
            if 1 == 1:  ## if there is a pattern format : 3-2-3 or 2-3-2
            ## the pattern format: 3-2-3 or 1-2-1   there is a local minial  points 
            #    max_position = [i for i, j in enumerate(areas_index) if j == max_value]
                hopfield_bit = 0

                if type == 0 or type == 1:
                    ascent_bit = 0
                    swing_bit = 0
                    sense_range = 1
                    sense_flag = 0
                    
                    ascent_threshold =  1 
                    swing_threshold = 2 
                    
                                    
                    for i in range(0,len(areas_index)): ## loop in frame
                        sense_flag_as = 0
                        sense_flag_sw = 0
                        j = 1
                        max_value = 100
                        min_value = 0 
                        while sense_flag_as == 0 and sense_flag_sw == 0: ## loop in sense range
                            if i+j +1 > len(areas_index)-1:
                                sense_flag_as = 1
                                sense_flag_sw = 1
                            else:
                                if areas_index[i] != -1 and areas_index[i+j] != -1 and areas_index[i+j+1] != -1:
                                    #if areas_index[i] > areas_index[i+j]  and areas_index[i] < max_value: ## detect pattern 3-2-X-X-1
                                    if areas_index[i] > areas_index[i+j]  and areas_index[i+j] > areas_index[i+j+1]:
                                            direction_flow = 'top->down'
                                            if direction_flow == spike_flow:
                                                ascent_bit = ascent_bit +1
                                                sense_flag_as = 1
                                                #max_value = areas_index[i+j]
                                    elif areas_index[i] < areas_index[i+j] and areas_index[i+j] < areas_index[i+j+1]: 
                                            direction_flow = 'bottom->up'
                                            if direction_flow == spike_flow: 
                                                ascent_bit = ascent_bit +1 
                                                sense_flag_as = 1 
                                                min_value = areas_index[i]                                              
                                    else:
                                        pass
                                    ## detect swing bit
                                    if (areas_index[i] > areas_index[i+j]) and (areas_index[i+j]< areas_index[i+j+1]): #5
                                        direction_flow = 'left->right' 
                                        #if direction_flow == spike_flow:
                                        swing_bit = swing_bit +1
                                        sense_flag_sw = 1
                                        max_value = areas_index[i]
                                    elif (areas_index[i] < areas_index[i+j]) and (areas_index[i+j]> areas_index[i+j+1]): #5
                                        direction_flow = 'right->left'
                                        #if direction_flow == spike_flow: 
                                        swing_bit = swing_bit +1 
                                        sense_flag_sw = 1  
                                        min_value = areas_index[i]                                               
                                    else:
                                        pass


                            if j >= sense_range:
                                sense_flag_sw = 1
                                sense_flag_as = 1
                            else:
                                j= j+1     
                    # print('ascent_bit:', ascent_bit)
                    # print('swing_bit:',swing_bit)


                    if ascent_bit >= ascent_threshold:
                        feature_ascent = 1                                              
                    else:
                        feature_ascent = 0

                    if swing_bit >= swing_threshold:
                        feature_swing = 1                                              
                    else:
                        feature_swing = 0    

                elif type == 2 or type == 3: 
                    ascent_bit = 0
                    swing_bit = 0
                    sense_range = 1
                    sense_flag = 0
                    swing_threshold = 2
                    ascent_threshold = 1     
                    for i in range(0,len(areas_index)): ## loop in frame
                        sense_flag_as = 0
                        sense_flag_sw = 0
                        j = 1
                        max_value = 100
                        min_value = 0 
                        while sense_flag_as == 0 and sense_flag_sw == 0: ## loop in sense range
                            if i+j+1 > len(areas_index)-1:
                                sense_flag_as = 1
                                sense_flag_sw = 1
                            else:
                                if areas_index[i] != -1 and areas_index[i+j] != -1 and areas_index[i+j+1] != -1:
                                    if (areas_index[i] > areas_index[i+j]) and (areas_index[i+j]< areas_index[i+j+1]): #5
                                        direction_flow = 'left->right' 
                                        #if direction_flow == spike_flow:
                                        swing_bit = swing_bit +1
                                        sense_flag_sw = 1
                                        #max_value = areas_index[i]
                                    elif (areas_index[i] < areas_index[i+j]) and (areas_index[i+j]> areas_index[i+j+1]): #5
                                        direction_flow = 'right->left'
                                        #if direction_flow == spike_flow: 
                                        swing_bit = swing_bit +1 
                                        sense_flag_sw = 1  
                                        #min_value = areas_index[i]                                               
                                    else:
                                        pass

                                if areas_index[i] != -1 and areas_index[i+j] != -1 and areas_index[i+j+1] != -1:
                                    #if areas_index[i] > areas_index[i+j]  and areas_index[i] < max_value: ## detect pattern 3-2-X-X-1
                                    if areas_index[i] > areas_index[i+j]  and areas_index[i+j] > areas_index[i+j+1]: ## detect pattern 3-2-X-X-1
                                            direction_flow = 'top->down'
                                            #if direction_flow == spike_flow:
                                            ascent_bit = ascent_bit +1
                                            sense_flag_as = 1
                                            #max_value = areas_index[i+j]
                                    elif areas_index[i] < areas_index[i+j] and areas_index[i+j] < areas_index[i+j+1]: 
                                            direction_flow = 'bottom->up'
                                            #if direction_flow == spike_flow: 
                                            ascent_bit = ascent_bit +1 
                                            sense_flag_as = 1 
                                            #min_value = areas_index[i]                                              
                                    else:
                                        pass                                    
                            if j >= sense_range:
                                sense_flag_sw = 1
                                sense_flag_as = 1
                            else:
                                j= j+1     
                    # print('ascent_bit:', ascent_bit)
                    # print('swing_bit:',swing_bit)

                    if ascent_bit >= ascent_threshold:
                        feature_ascent = 1                                              
                    else:
                        feature_ascent = 0

                    if swing_bit >= swing_threshold:
                        feature_swing = 1                                              
                    else:
                        feature_swing = 0       
                else:
                    pass                           
                #if hopfield_bit == 1:                                                                   
                #    print('direction_flow:',direction_flow)

        # print(feature_ascent,feature_swing)
        return areas_index, feature_ascent, feature_swing

    

    def unitC_space_expert1(self, testdata):
        # e1 = SGF_expert()
        self.thres_s3 = [int(t) for t in self.args.thres_s3.split('_')]
        self.thres_s4 = [int(t) for t in self.args.thres_s4.split('_')]
        self.thres_s5 = [int(t) for t in self.args.thres_s5.split('_')]

        space_neuron3 = self.expert_space(int(self.args.resolution_s3.split('_')[0]), int(self.args.resolution_s3.split('_')[1]), \
            testdata, self.thres_s3, [0,0], thres_bit=self.args.thres_bit_s3, \
            thres_step=self.args.thres_step_s3, thres_inc_factor=[1,-1], if_vote=self.args.vote_thres_step)

        if space_neuron3[1][0] == 0:
            space_neuron4_1 = self.expert_space(int(self.args.resolution_s4.split('_')[0]), int(self.args.resolution_s4.split('_')[1]), \
                testdata[27:,:,:], self.thres_s4, [0,0], thres_bit=self.args.thres_bit_s4, \
                thres_step=self.args.thres_step_s4, thres_inc_factor=[1,-1], if_vote=self.args.vote_thres_step)
            space_neuron4_2 = self.expert_space(int(self.args.resolution_s4.split('_')[0]), int(self.args.resolution_s4.split('_')[1]), \
                testdata[0:27,:,:], self.thres_s4, [0,0], thres_bit=self.args.thres_bit_s4, \
                thres_step=self.args.thres_step_s4, thres_inc_factor=[1,-1], if_vote=self.args.vote_thres_step)                                

            if space_neuron4_2[0][0] == 1:
                predict_event = "2"
            elif space_neuron4_1[0][0] == 1:
                predict_event = "10"
            else:
                predict_event = "10"
        else:
            predict_event = "1+8+9+10"
        # print(space_neuron3.flatten(), space_neuron4_1.flatten(), space_neuron4_2.flatten())
        return predict_event

    def unitC_space_expert2(self, testdata):
        testdata_sum = np.sum(testdata,0)      

        x, y = self.cal_center(testdata_sum[60:110,:,:])
        x, y = int(x), int(y)+60
        active_zone = testdata_sum[y-20:y+20, x-30:x+30, :]
        active_zone = np.where((active_zone>5)&(active_zone<40), 1, 0)

        thres1 = self.args.test5
        testdata_sum = testdata_sum[self.args.test1:self.args.test2,:,:]
        num_list = np.zeros(testdata_sum.shape[1])
        for x in range(testdata_sum.shape[1]):
            cnt = 0
            for y in range(testdata_sum.shape[0]):
                if testdata_sum[y][x] >= thres1:
                    cnt += testdata_sum[y][x]
            num_list[x] = cnt

        smooth_factor = self.args.test6
        num_list = convolve(num_list, np.ones(smooth_factor))/smooth_factor

        increase_list = list()
        skip = 1
        increase_flag = 0
        thres = self.args.test7
        for num_i in range(int(len(num_list[:-1]))):
            if num_list[(num_i+1)*skip] > num_list[num_i*skip]+thres:
                increase_flag = 1
            elif num_list[(num_i+1)*skip] < num_list[num_i*skip]-thres:
                increase_flag = -1
            increase_list.append(increase_flag)
        
        edge_list = list()
        for i_i in range(len(increase_list[:-2])):
            if increase_list[i_i+1] == 1 and increase_list[i_i] == -1:
                edge_list.append(1)
            elif increase_list[i_i+1] == -1 and increase_list[i_i] == 1:
                edge_list.append(1)
            else:
                edge_list.append(0)
        edge_num = sum(edge_list[self.args.test3:self.args.test4])
        
        if edge_num != 1:
            predict_event = "1+9+10"
        elif  active_zone.sum()<867:
            predict_event = "1+9+10"
        else:
            predict_event = "8"

        return predict_event

    def unitC_space_expert3(self, testdata):
        testdata[testdata < 0] = 1
        testdata_sum = np.sum(testdata,0) 
        x_c, y_c = self.cal_center(np.where(testdata_sum>10,1,0))
        x_c, y_c = int(x_c), int(y_c)
        x1, y1 = self.cal_center(np.where(testdata_sum[:,0:x_c,:]>10,1,0))
        x2, y2 = self.cal_center(np.where(testdata_sum[:,x_c:128,:]>10,1,0))
        delta_h = abs(y2-y1)
        # print(y1, y2, abs(y2-y1))
        if delta_h < 9:
            predict_event = "1+9"
        else:
            predict_event = "10"

        return predict_event

    def cal_center(self, input_img):
        cnt = 0
        x_acc = 0
        y_acc = 0
        for y in range(input_img.shape[0]):
            for x in range(input_img.shape[1]):
                if input_img[y][x] > 0:
                    cnt += 1
                    x_acc += x
                    y_acc += y
        x_center = x_acc / cnt
        y_center = y_acc / cnt
        return [x_center, y_center]
