import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import cfg
class Visualization(object):
    def __init__(self, video):
      self.dataset = video

    def generateDVS(self): 
        
      for id in range(0,np.shape(self.dataset)[3]):       
        data_id = str(id)+'.avi'
        OUTPUT_FILE = os.path.join(cfg.code_path, data_id) 
        start_frame = 0
        end_frame = np.shape(self.dataset)[0]
        width = 128#int(540/2)
        height = 128 #int(420/2)
        writer = cv2.VideoWriter(OUTPUT_FILE, 
              cv2.VideoWriter_fourcc('I', '4', '2', '0'),
              10, # fps
              (width,height )) # resolution
        have_more_frame = True
        c = 0   
        while have_more_frame: 
         
         c += 1
         if c>= start_frame and c<= end_frame-1:
      #  cv2.waitKey(1);,
            gray_frame = np.full(( np.shape(self.dataset)[1],np.shape(self.dataset)[2], 3),0)
            #gray_frame[:,:,2] = self.dataset[c,:,:,id]
            gray_frame[:,:,1]  = self.dataset[c,:,:,id]*100
            #gray_frame[:,:,0]  = self.dataset[c,:,:,id]
            writer.write(np.uint8(gray_frame))
            
        #print(str(c) + ' is ok')
         if c>end_frame:
            print('completely!')
            break         

    def generate_picture(self,event_id, frame_id, name):  
              self.event_id = event_id
              self.frame_id = frame_id
              self.name = name
              target = np.full((np.shape(self.dataset)[1],np.shape(self.dataset)[2], 3),0)
              target[:,:,1] = self.dataset[self.frame_id,:,:,self.event_id]*100
              fig, ax = plt.subplots()
              imgplot = ax.imshow(target)  
              plt.show()
              plt.savefig(str(name))   
              

    def generate_neuronout(self, V, x_id, y_id, event_id,name):
              self.V = V
              self.x_id = x_id
              self.y_id = y_id
              self.event_id = event_id
              fig, ax = plt.subplots()
              ax.plot(self.V[:,self.x_id,self.y_id,self.event_id])
              plt.show()
              plt.savefig(str(name)) 

    def generate_spike(self, V, x_id, y_id, event_id,name):
              self.V = V
              self.x_id = x_id
              self.y_id = y_id
              self.event_id = event_id
              fig, ax = plt.subplots()
              ax.plot(self.V[:,self.x_id,self.y_id,self.event_id],'ro')
              plt.show()
              plt.savefig(str(name)) 

    def generate_st(self):             
              for i in range(0, np.shape(self.dataset)[2]):
                name = "event" + str(i)
                self.event_id = i
                X = np.arange(0, np.shape(self.dataset)[0]+1,1)
                Y = np.arange(0, np.shape(self.dataset)[1]+1,1)
                Z = self.dataset[:,:,self.event_id]
                fig, ax = plt.subplots()
                cmap = plt.get_cmap('PiYG')
                levels = matplotlib.ticker.MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
                norm = matplotlib.colors.BoundaryNorm(levels, ncolors = cmap.N, clip=True)
                im = ax.pcolormesh(Y, X, Z, cmap=cmap, norm=norm)
                fig.colorbar(im, ax=ax)                            
                plt.show()
                plt.savefig(str(name))

    def generate_st_event(self):
              for i in range(0, np.shape(self.dataset)[0]):
                name = "single_event" + str(i)
                self.event_frame = i
                X = np.arange(0, np.shape(self.dataset)[1]+1,1)
                Y = np.arange(0, np.shape(self.dataset)[2]+1,1)
                Z = self.dataset[self.event_frame,:,:]
                fig, ax = plt.subplots()
                cmap = plt.get_cmap('PiYG')
                levels = matplotlib.ticker.MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())
                norm = matplotlib.colors.BoundaryNorm(levels, ncolors = cmap.N, clip=True)
                im = ax.pcolormesh(Y, X, Z, cmap=cmap, norm=norm)
                fig.colorbar(im, ax=ax)                            
                plt.show()
                plt.savefig(str(name))                        
      

    def generate_connect(self):
            for i in range(0, np.shape(self.dataset)[1]):
               name = "connect" + str(i)
               target = np.reshape(self.dataset[:,i],(128,128))
               fig, ax = plt.subplots()
               imgplot = ax.imshow(target)  
               plt.show()
               plt.savefig(str(name))

    def generate_structure_pattern(self):
            for i in range(0, np.shape(self.dataset)[3]):
               target = self.dataset[:,:,:,i]
               target_1d = np.reshape(target, (np.shape(self.dataset)[0], np.shape(self.dataset)[1]*np.shape(self.dataset)[2]))
               name = "struct" + str(i) 
               fig, ax = plt.subplots()         
               imgplot = ax.imshow(np.transpose(target_1d)) 
               plt.show()
               plt.savefig(str(name))

    def generate_stbar(self):
            st_index = [i for i in range (np.shape(self.dataset)[0])]
            for i in range(0, np.shape(self.dataset)[1]):
                name = "bar" + str(i) 
                fig, ax = plt.subplots()
                ax.bar(st_index, self.dataset[:,i]) 
                plt.show() 
                plt.savefig(str(name))

    def generate_weight_hist(self,sigma,mu):
            #name = "weight"
            #fig, ax = plt.subplots()
            #ax.plot(self.dataset[1,:])
            #plt.show()
            #plt.savefig(str(name))
            name1 = "hist"
            count, bins, ignored = plt.hist(self.dataset[1,:], 10, density=True)
            #plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
            plt.show()
            plt.savefig(str(name1))

    #def generate_weight_integral(self)        
    def generate_temporal_profiling(self, stim, syn, temporal_output):
            fig, ax = plt.subplots()
            for i in range(0, np.shape(stim)[0]):
                for j in range(0, np.shape(stim)[1]):
                    if stim[i,j] == 1:
                       ax.plot(i, j, 'o', color='black',markersize=5)

            for i in range(0, np.shape(syn)[0]):
                for j in range(0, np.shape(syn)[1]):
                    if syn[i,j] == 1:
                       ax.plot(i, j, 's', color='blue',markersize=5)

            for i in range(0, np.shape(temporal_output)[0]):
                for j in range(0, np.shape(temporal_output)[1]):
                    if temporal_output[i,j] == 1:
                       ax.plot(i, j, 'd', color='red',markersize=5)

            plt.xlim(0, 127)
            plt.ylim(0, 127)
            plt.show()

            plt.savefig('temporal_profilling')     

    def generate_weight_map(self,weight, start, end):
            fig, ax = plt.subplots()
            for i in range(start, end):
              value = weight[:,:,i]
              for j in range(0, np.shape(value)[0]):
                for k in range(0, np.shape(value)[1]):
                    if value[j,k] == 1:
                       ax.plot(k, j, '+', color='black',markersize=5)
                      
            #plt.xlim(0, 127)
            #plt.ylim(0, 127)
            plt.show()

            plt.savefig('weight_map')                            




                  



              
