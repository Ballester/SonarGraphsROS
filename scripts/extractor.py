import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import sqrt, atan, degrees
from time import sleep
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from matplotlib.patches import Ellipse

class GaussianFeatureExtractor(object):
    
    def __init__(self):
        #Parameter inicialization
        self.all_graphs = open('all_graphs.txt', 'w')   #Stores all the graphs info
        self.threshold = 1                              #Minimum std deviation for storing a node
        self.edge_threshold = 50                        #Minimum distance for storing a line between nodes
        self.merge_thresh = 3                           #Similarity of std deviation for merging nodes
        self.filter_min = 0                             #Minimum value for thresholding filter
        self.filter_max = 255                           #Maximum value for thresholding filter
        self.filter_pixel_lin = 2000                    #Number of highest-intensity pixel who we will evaluate in PDI stage
        self.jumper_i = 40                              #Number of pixels in i
        self.jumper_j = 40                              #Number of pixels in j
        self.img_height = 781                           #Size of the input image
        self.img_width = 1430                           #Size of the input image  
        
        #Control Variables
        self.bridge = CvBridge()                        #Bridge between cv and ros images
        self.jump_i = self.img_height/self.jumper_i     #Window size in i
        self.jump_j = self.img_width/self.jumper_j      #Window size in j
        
    #Returns a ROS image from a opencv image
    def convertImage(self, image):
        return self.bridge.cv2_to_imgmsg(image, "mono8")
        
    #Returns the original image and the initiliazed image (with mask and threshold)
    def initImage(self, data):
        #creating lists
        self.centers_x = []
        self.centers_y = []
        self.std_x = []
        self.std_y = []
        
        #Convert image from ROS msg to OpenCV
        img = self.bridge.imgmsg_to_cv2(data, "mono16")
        
        # Apply a normalization in mono16 image for convert to 8bits image
        self.img_norm = np.empty([img.shape[0] , img.shape[1]], np.uint16)
        cv2.normalize(img, self.img_norm, 0 , 255, cv2.NORM_MINMAX)
        
        # Convert uint16 to uint8 gray_scale depth
        self.img_norm = np.uint8(self.img_norm)

        hist = cv2.calcHist([self.img_norm],[0],None, [256] , [ 0, 255])
        
        # Calculate auto threshold
        sum = 0
        threshold = 255
        for pCount in hist[::-1]:
            if sum < self.filter_pixel_lin:
                sum += pCount
                threshold-= 1
            else:
                break
             
        #Thresholding the image and grayscaling
        self.bin_img = cv2.threshold(self.img_norm, threshold, 255 ,cv2.THRESH_BINARY)[1]

        #Setting up the mask
        mask = self.bin_img.copy()
        for i in range(0, 490):
            for j in range(173, 1174):
                mask[i][j] = 0
             
        #Applying the mask
        self.bin_img = self.bin_img - mask
        return img, self.bin_img
        
    #Returns gaussian image
    #Creates the segments and allocates the centers and standard deviations
    def createSegments(self, save_gaussian_image=False, gaussian_image_name=''):
        print "Creating segments!"
        
        #Initializing values for the segmentation
        last_i = 0
        last_j = 0
        
        #Initializing blank image
        gaussian_image = np.zeros((self.jumper_i+1, self.jumper_j+1, 1), np.uint8)
        pos_i = 0
        pos_j = 0
        
        self.std_x = []
        self.std_y = []
                
        #print self.bin_img    
        #Goes through segments and get values
        while last_j < self.img_width - (self.jump_j) - 1:
            while last_i < self.img_height - (self.jump_i) - 1:
                valuex = []
                valuey = []
                
                              
                i = last_i
                #end_i = last_i + self.jump_i
                #j = last_j
                #end_j = last_j + self.jump_j
                
                '''
                while i <= last_i + self.jump_i-1:
                    j = last_j
                    while j <= last_j + self.jump_j-1:                        
                        if self.bin_img[i][j] > 0:
                            valuex.append(i)
                            valuey.append(j)
                        j+=1
                    i+=1
                
                '''
                for i in range(last_i, last_i + self.jump_i):
                    for j in range(last_j, last_j + self.jump_j):
                        if self.bin_img[i][j] > 0:
                            valuex.append(i)
                            valuey.append(j)

                if valuex != []:
                    std_x = np.std(valuex)
                    center_x = np.mean( valuex )
                    
                if valuey != []:
                    std_y = np.std(valuey)
                    center_y = np.mean( valuey )
                
                if valuex != [] and valuey != []:
                    self.std_x.append(np.std(valuex))
                    self.std_y.append(np.std(valuey))
                    self.centers_x.append(np.mean( valuex ) )
                    self.centers_y.append(np.mean( valuey ) )
                    #print valuex
                    #print valuey
                    print (self.centers_x[-1], self.centers_y[-1]), (self.std_x[-1], self.std_y[-1]) 
                            
                last_i += self.jump_i
                
                #print "standart derivation = " , self.std_x[-1] , self.std_y[-1]
                
                if save_gaussian_image:
                    if valuex != [] and valuey != []:
                        gaussian_image[pos_i][pos_j] = (np.std(valuex) + np.std(valuey))
                    else:
                        gaussian_image[pos_i][pos_j] = (0)
                    
                    pos_i += 1
                
            last_i = 0
            last_j += self.jump_j
            pos_i = 0
            pos_j += 1
        
        if save_gaussian_image:
            #gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_RGB2GRAY)
            gaussian_image = cv2.normalize(gaussian_image, gaussian_image, 0, 255, cv2.NORM_MINMAX)
            
            cv2.imwrite(gaussian_image_name, gaussian_image)


        #gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2GRAY)
        return gaussian_image
    
    def createSegmentsFloodFill(self):
        print "Creating segments!"
        
        #Initializing values for the segmentation
        last_i = 0
        last_j = 0
        
        #Initializing blank image
        gaussian_image = np.zeros((self.jumper_i+1, self.jumper_j+1, 1), np.uint8)
        pos_i = 0
        pos_j = 0
        
        self.std_x = []
        self.std_y = []
                
        #print self.bin_img    
        #Goes through segments and get values
        while last_j < self.img_width - (self.jump_j) - 1:
            while last_i < self.img_height - (self.jump_i) - 1:
                valuex = []
                valuey = []
                
               
                i = last_i
                #end_i = last_i + self.jump_i
                #j = last_j
                #end_j = last_j + self.jump_j
                
                '''
                while i <= last_i + self.jump_i-1:
                    j = last_j
                    while j <= last_j + self.jump_j-1:                        
                        if self.bin_img[i][j] > 0:
                            valuex.append(i)
                            valuey.append(j)
                        j+=1
                    i+=1
                
                '''
                for i in range(last_i, last_i + self.jump_i):
                    for j in range(last_j, last_j + self.jump_j):
                        if self.bin_img[i][j] > 0:
                            valuex.append(i)
                            valuey.append(j)

                if valuex != []:
                    std_x = np.std(valuex)
                    center_x = np.mean( valuex )
                    
                if valuey != []:
                    std_y = np.std(valuey)
                    center_y = np.mean( valuey )
                
                if valuex != [] and valuey != []:
                    self.std_x.append(np.std(valuex))
                    self.std_y.append(np.std(valuey))
                    self.centers_x.append(np.mean( valuex ) )
                    self.centers_y.append(np.mean( valuey ) )
                    #print valuex
                    #print valuey
                    print (self.centers_x[-1], self.centers_y[-1]), (self.std_x[-1], self.std_y[-1]) 
                            
                last_i += self.jump_i
                
                #print "standart derivation = " , self.std_x[-1] , self.std_y[-1]
                
                if save_gaussian_image:
                    if valuex != [] and valuey != []:
                        gaussian_image[pos_i][pos_j] = (np.std(valuex) + np.std(valuey))
                    else:
                        gaussian_image[pos_i][pos_j] = (0)
                    
                    pos_i += 1
                
            last_i = 0
            last_j += self.jump_j
            pos_i = 0
            pos_j += 1
        
        if save_gaussian_image:
            #gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_RGB2GRAY)
            gaussian_image = cv2.normalize(gaussian_image, gaussian_image, 0, 255, cv2.NORM_MINMAX)
            
            cv2.imwrite(gaussian_image_name, gaussian_image)


        #gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2GRAY)
        return gaussian_image
    
    #Returns the graph image    
    def drawGraph(self, merge=False, write=False, export_name=''):
        #Initializing lists
        centers_used = []
        angles = []
        ellipses = []
        lines = []
        
        #Control variables
        n = -1   #number of ellipses
        
        print "Drawing Graph!!"
        
        #Export variables
        sum_std_x = 0
        sum_std_y = 0
        n_gaussians = 0
        n_edges = 0
        
        # graph_img = np.empty(self.img_norm.shape , dtype = np.uint8)
        graph_img = cv2.cvtColor(self.bin_img, cv2.COLOR_GRAY2BGR)
        
        #Save ellipses if they are good enough
        for i in range(0, len(self.centers_x)):
            #print "Center", self.centers_x[i] , self.centers_y[i] , "std" , self.std_x[i], self.std_y[i]
            if self.std_x[i] > 0 and self.std_y[i] > 0:
                    
                sum_std_x += self.std_x[i]
                sum_std_y += self.std_y[i]
                n_gaussians += 1
            
                axes = (int((self.std_x[i]*sqrt(5.991))), int((self.std_y[i]*sqrt(5.991))))
                angle = atan((self.std_y[i]**2)/(self.std_x[i]**2))
                ellipses.append((self.centers_y[i], self.centers_x[i], axes, angle))
                
                n += 1
                
                #Verify if it should merge with other
                if merge:
                    for j in range(0, len(ellipses)-1):
                        if ellipses[j] != -1 and ellipses[n] != -1:
                            if sqrt((ellipses[j][0] - ellipses[n][0])**2 + (ellipses[j][1] - ellipses[n][1])**2) < self.edge_threshold:
                                if abs((ellipses[j][2][0] + ellipses[j][2][1]) - (ellipses[n][2][0] + ellipses[n][2][1])) < self.merge_thresh:
                                    ellipses.append(((ellipses[j][0] + ellipses[n][0])/2, (ellipses[j][1] + ellipses[n][1])/2, ((ellipses[j][2][0] + ellipses[n][2][0]), ellipses[j][2][1] + ellipses[n][2][1]), (ellipses[j][3] + ellipses[n][3])/2))
                                    ellipses[j] = -1
                                    ellipses[n] = -1
            #else:
            #    print "Low std!",  self.std_x[i] , self.std_y[i]
                
                    
        #Save lines by euclidian distance
        for i in range(0, len(ellipses)-1):
            for j in range(i+1, len(ellipses)):
                if ellipses[j] != -1 and ellipses[i] != -1:
                    if sqrt((ellipses[j][0] - ellipses[i][0])**2 + (ellipses[j][1] - ellipses[i][1])**2) < self.edge_threshold:
                        lines.append(((int(ellipses[i][0]), int(ellipses[i][1])), (int(ellipses[j][0]), int(ellipses[j][1])), (255,0,0)))
                        n_edges += 1
        
        #Draw ellipses
        for el in ellipses:
            if el != -1:
                cv2.ellipse(graph_img, (int(el[0]), int(el[1])), el[2], degrees(el[3]), 0 , 360, (255,0,0),  2)
                

        #Draw lines
        for line in lines:
            cv2.line(graph_img, line[0], line[1], (255,0,0))
        
        #Write to file
        if write:           
            self.all_graphs.write(str(sum_std_x)+' ')
            self.all_graphs.write(str(sum_std_y)+' ')
            self.all_graphs.write(str(n_gaussians)+' ')
            self.all_graphs.write(str(n_edges)+'\n')
        
        return graph_img
                  
    
    #Plots and saves the images                
    def plotImages(self, img, save_plot, plot_name, save_graph=False, graph_name='', show_plot=False):
        plt.subplot(1,2,2)
        plt.title('Thresholded')
        plt.imshow(self.bin_img)
        plt.subplot(1,2,1)
        plt.title('Original')
        plt.imshow(img)
        if save_plot:
            plt.savefig(plot_name)
        if save_graph:
            cv2.imwrite(graph_name, self.bin_img)    
        if show_plot:
            plt.show()
                
    

    
    
