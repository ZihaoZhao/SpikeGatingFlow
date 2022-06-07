import numpy as np


class SGF_prior_knowledge(object):
    def __init__(self):
        super().__init__()           


    def prior_knowledge(self, actions, location, timings):
        ## actions: the detected movement in an actions
        ##          [0,1,2,3] #0: top->down; 1:botom->up; 2:left->right; 3: right->left 
        ## location: the movement happened areas
        ##          [0,1,2,3] #0: horonizital right area(>7)
        #                     #1: horonizital left area(<=7)
        #                     #2: vertical top area(>5)
        #                     #3: vertical bottom area(<=5)
        ## timings: the movement happened timing

        ##---------------------------------------##  
                            #      ----------》(2)      
                            #     ^ (1)      |          clock wise: (3->1)-[XXX1],(1->2)-[XX1X],(2->0)-[XXX1], (0->3)=[XXX1]
                            #     |          |           two movements happened same time:  (0,2)=[xx10]
                            #     |          X (0)
                            #     (3)<----------    
        ##---------------------------------------##    

        ##---------------------------------------##  
                            #      <----------(3)      
                            #      (0)       ^
                            #     |          |          anti-clock wise: (1,3),(3,0),(0,2), (2,1)
                            #     |          |       
                            #     |          |
                            #     X           (1)
                            #     (2)----------》    
        ##---------------------------------------##    


        ## decoding the timing activities

        if np.all(actions) == 1: ### it must be either clock or anti-clock [ 1111]
            print('all events happen.') 
            timing_origion = timings.copy()
            timings.sort()
            min_frame = min(timings)
            first = [i for i, j in enumerate(timing_origion) if j == min_frame]

            for i in range(0,len(timings)):
                if timings[i] == min_frame:
                    timings[i] = 100
            min_second_frame =  min(timings)       
            second = [i for i, j in enumerate(timing_origion) if j == min_second_frame]

            final = [0 for i in range(0,2)]

            if len(first) == 1 and len(second) == 1 : ## if the movement sequence is  [7,3,X,X]
                test_list = [first,second]
                if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                    ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                    final[1] = 1
                elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                    final[0] = 1 
                else:
                    pass
            
            elif len(first) != 1 and len(second) == 1 and second != 100: ## [1,1,2,3]
                finish_flag = 0
                for i in range(0,len(first)):
                    if finish_flag == 0:
                        test_list = [[first[i]],second]
                        if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                            ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                            final[1] = 1
                            finish_flag = 1
                        elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                            final[0] = 1
                            finish_flag = 1 
                        else:
                            pass        

            elif len(first) == 1 and len(second) != 1 :   ## [1,2,2,3]     
                for i in range(0,len(second)):
                    test_list = [first,[second[i]]]
                    if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                        ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                        final[1] = 1
                    elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                        final[0] = 1 
                    else:
                        pass  
            
            else:
                pass                        
        
        else:  ## [X000]  
            final = [0 for i in range(0,2)]
            timing_origion = timings.copy()
            timings.sort()
            min_frame = min(timings)
            first = [i for i, j in enumerate(timing_origion) if j == min_frame]
            if len(first) == 2:    ## if two events happened in the same time [1,1,0,X]
                '''
                print('two actions happened in the same time points')

                if (first[0] ==0 and first[1]  == 3) or (first[0] ==3 and first[1] == 0):
                    if location[2] == 1:  
                        final[0] = 1
                    else:    
                        final[1] = 1
                elif (first[0] == 0  and first[1] == 2) or (first[0] ==2 and first[1] == 0):
                    if location[2] == 1:  
                        final[1] = 1
                    else:     
                        final[0] = 1
                elif (first[0] ==1 and first[1] == 3) or (first[0] ==3 and first[1] == 1):  
                    if location[11] == 1:
                        final[1] = 1
                    else:
                        final[0] = 1
                elif (first[0] ==1 and first[1] == 2) or (first[0] ==2 and first[1] == 1):  
                    if location[11] == 1:
                        final[0] = 1   
                    else:
                        final[1] = 0 
                '''
                location_flag = 1
                finish_flag = 0
                for i in range(0,len(timings)):
                    if timings[i] == min_frame:
                        timings[i] = 100
                min_second_frame =  min(timings)       
                second = [i for i, j in enumerate(timing_origion) if j == min_second_frame]  ## check the second event           
                test_list = [[first[0]],second ]
                if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                    ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                    final[1] = 1
                    location_flag = 0
                    finish_flag = 1
                elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                    final[0] = 1 
                    location_flag = 0
                    finish_flag = 1
                else:
                    pass     

                if finish_flag == 0:
                    test_list = [[first[1]],second ]
                    if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                        ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                        final[1] = 1
                        location_flag = 0
                    elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                        final[0] = 1 
                        location_flag = 0 
                    else:
                        if location_flag == 1:
                            if (first[0] ==0 and first[1]  == 3) or (first[0] ==3 and first[1] == 0):
                                if location[2] == 1:  
                                    final[0] = 1
                                else:    
                                    final[1] = 1
                            elif (first[0] == 0  and first[1] == 2) or (first[0] ==2 and first[1] == 0):
                                if location[2] == 1:  
                                    final[1] = 1
                                else:     
                                    final[0] = 1
                            elif (first[0] ==1 and first[1] == 3) or (first[0] ==3 and first[1] == 1):  
                                if location[11] == 1:
                                    final[1] = 1
                                else:
                                    final[0] = 1
                            elif (first[0] ==1 and first[1] == 2) or (first[0] ==2 and first[1] == 1):  
                                if location[11] == 1:
                                    final[0] = 1   
                                else:
                                    final[1] = 0  
                            else:
                                print('unknown issue1')                                                                
            elif len(first) == 1 and first != 100:   ## if there are events in sequence   [1,2,0,X] 
                for i in range(0,len(timings)):
                    if timings[i] == min_frame:
                        timings[i] = 100
                min_second_frame =  min(timings)       
                second = [i for i, j in enumerate(timing_origion) if j == min_second_frame]  ## check the second event
                if len(second) == 1 and timing_origion[second[0]] != 100:  #[1,0,2,X] 
                    '''
                    if (first ==[0] and second == [3]) or (first ==[3 ]and second == [0]):
                        if location[2] == 1:  
                            final[0] = 1
                        else:    
                            final[1] = 1
                    elif (first ==[0] and second == [2]) or (first ==[2] and second == [0]):
                        if location[2] == 1:  
                            final[1] = 1
                        else:     
                            final[0] = 1
                    elif (first ==[1] and second == [3]) or (first ==[3] and second == [1]):  
                        if location[11] == 1:
                            final[1] = 1
                        else:
                            final[0] = 1
                    elif (first ==[1] and second == [2]) or (first ==[2] and second == [1]):  
                        if location[11] == 1:
                            final[0] = 1   
                        else:
                            final[1] = 1 
                    '''
                    test_list = [first,second]
                    if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                        ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                        final[1] = 1
                    elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                        final[0] = 1                          
                    else: #[1,2,0,X] 
                        print('system info: the two events are 0 and 1  or 2 and 3') 
                        for i in range(0,len(timings)):
                            if timings[i] == min_second_frame:
                                timings[i] = 100
                        min_third_frame =  min(timings)       
                        third = [i for i, j in enumerate(timing_origion) if j == min_third_frame]  ## 
                        if len(third) == 1 and third != 100 :
                            '''
                            if (first ==[0] and third == [3]) or (first ==[3 ]and third == [0]):
                                if location[2] == 1:  
                                    final[0] = 1
                                else:    
                                    final[1] = 1
                            elif (first ==[0] and third == [2]) or (first ==[2] and third == [0]):
                                if location[2] == 1:  
                                    final[1] = 1
                                else:     
                                    final[0] = 1
                            elif (first ==[1] and third == [3]) or (first ==[3] and third == [1]):  
                                if location[11] == 1:
                                    final[1] = 1
                                else:
                                    final[0] = 1
                            elif (first ==[1] and third == [2]) or (first ==[2] and third == [1]):  
                                if location[11] == 1:
                                    final[0] = 1   
                                else:
                                    final[1] = 1 
                            '''
                            test_list = [first,third]
                            if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                                ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                                final[1] = 1
                            elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                                final[0] = 1                                     
                            else:
                                print('system error2: complex situations')                                                               
                elif len(second) >1 and timing_origion[second[0]] != 100: #[1,2,2,0] 
                        finish_flag1 = 0
                        test_list = [first,[second[0]] ]
                        if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                            ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                            final[1] = 1
                            finish_flag1 = 1
                        elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                            final[0] = 1 
                            finish_flag1 = 1
                        else:
                            pass     

                        if finish_flag1 == 0:
                            test_list = [first,[second[1]] ]
                            if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                                ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                                final[1] = 1
                            elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                                final[0] = 1  
                            else:
                                pass                                
                        '''
                    if (first ==[0] and second[0] == 3) or (first ==[3] and second[0] == 0):
                        if location[2] == 1:  
                            final[0] = 1
                        else:    
                            final[1] = 1
                    elif (first ==[0] and second[0] == 2) or (first ==[2] and second[0] == 0):
                        if location[2] == 1:  
                            final[1] = 1
                        else:     
                            final[0] = 1
                    elif (first ==[1] and second[0] == 3) or (first ==[3] and second[0] == 1):  
                        if location[11] == 1:
                            final[1] = 1
                        else:
                            final[0] = 1
                    elif (first ==[1] and second[0] == 2) or (first ==[2] and second[0] == 1):  
                        if location[10] == 1:
                            final[0] = 1   
                        else:
                            final[1]=   1 

                    if (first ==[0] and second[1] == 3) or (first ==[3] and second[1] == 0):
                        if location[2] == 1:  
                            final[0] = 1
                        else:    
                            final[1] = 1
                    elif (first ==[0] and second[1] == 2) or (first ==[2] and second[1] == 0):
                        if location[2] == 1:  
                            final[1] = 1
                        else:     
                            final[0] = 1
                    elif (first ==[1] and second[1] == 3) or (first ==[3] and second[1] == 1):  
                        if location[11] == 1:
                            final[1] = 1
                        else:
                            final[0] = 1
                    elif (first ==[1] and second[1] == 2) or (first ==[2] and second[1] == 1):  
                        if location[10] == 1:
                            final[0] = 1   
                        else:
                            final[1]=    1 
                        '''                                                                                                  
                else:
                    # print('system error4: unknown')
                    a = 1
                    #if np.shape(np.nonzero(actions))[0] == 1: ## special case
                    #    final[0] = 1 
                    pass  
            elif len(first) == 3:
                #if location[5] == 1:
                #    final[1] = 1 
                #else:     
                #    final[0] = 1 
                pass
        return  final       


    def clockwise_knowledge(self, actions, location, timings):
        ## actions: the detected movement in an actions
        ##          [0,1,2,3] #0: top->down; 1:botom->up; 2:left->right; 3: right->left 
        ## location: the movement happened areas
        ##          [0,1,2,3] #0: horonizital right area(>7)
        #                     #1: horonizital left area(<=7)
        #                     #2: vertical top area(>5)
        #                     #3: vertical bottom area(<=5)
        ## timings: the movement happened timing

        ##---------------------------------------##  
                            #      ----------》(2)      
                            #     ^ (1)      |          clock wise: (3->1)-[XXX1],(1->2)-[XX1X],(2->0)-[XXX1], (0->3)=[XXX1]
                            #     |          |           two movements happened same time:  (0,2)=[xx10]
                            #     |          X (0)
                            #     (3)<----------    
        ##---------------------------------------##    

        ##---------------------------------------##  
                            #      <----------(3)      
                            #      (0)       ^
                            #     |          |          anti-clock wise: (1,3),(3,0),(0,2), (2,1)
                            #     |          |       
                            #     |          |
                            #     X           (1)
                            #     (2)----------》    
        ##---------------------------------------##    


        ## decoding the timing activities

        if np.all(actions) == 1: ### it must be either clock or anti-clock [ 1111]
            # print('all events happen.') 
            timing_origion = timings.copy()
            timings.sort()
            min_frame = min(timings)
            first = [i for i, j in enumerate(timing_origion) if j == min_frame]

            for i in range(0,len(timings)):
                if timings[i] == min_frame:
                    timings[i] = 100
            min_second_frame =  min(timings)       
            second = [i for i, j in enumerate(timing_origion) if j == min_second_frame]

            final = [0 for i in range(0,2)]

            if len(first) == 1 and len(second) == 1 : ## if the movement sequence is  [7,3,X,X]
                test_list = [first,second]
                if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                    ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                    final[1] = 1
                elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                    final[0] = 1 
                else:
                    pass                        
            
        else:  ## [XX00]  
            final = [0 for i in range(0,2)]
            timing_origion = timings.copy()
            timings.sort()
            min_frame = min(timings)
            first = [i for i, j in enumerate(timing_origion) if j == min_frame]
            if len(first) == 2:    ## if two events happened in the same time [1,1,0,X]
                pass                                                               
            elif len(first) == 1 and first != 100:   ## if there are events in sequence   [1,2,0,X] 
                for i in range(0,len(timings)):
                    if timings[i] == min_frame:
                        timings[i] = 100
                min_second_frame =  min(timings)       
                second = [i for i, j in enumerate(timing_origion) if j == min_second_frame]  ## check the second event
                if len(second) == 1 and timing_origion[second[0]] != 100:  #[1,0,2,X]  
                    test_list = [first,second]
                    if test_list ==[[3],[1]] or test_list ==[[1],[2]] or test_list ==[[2],[0]] or test_list ==[[0],[3]]:
                        ## from the sequence prior knowledge, it belongs to clock, check the location characteristics
                        final[1] = 1
                    elif test_list ==[[1],[3]] or test_list ==[[3],[0]] or test_list ==[[0],[2]] or test_list ==[[2],[1]]: 
                        final[0] = 1                          
                    else: #[1,2,0,X] 
                        pass                                                                   
                else:
                    # print('system error4: unknown')
                    #if np.shape(np.nonzero(actions))[0] == 1: ## special case
                    #    final[0] = 1 
                    pass  
            elif len(first) == 3:
                #if location[5] == 1:
                #    final[1] = 1 
                #else:     
                #    final[0] = 1 
                pass
        return  final                                                 
