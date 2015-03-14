import numpy as np
from math import sqrt, atan, degrees
from time import sleep
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2

class GaussianFeatureExtractor(object):
    
    def __init__(self):
        #Parameter inicialization
        self.all_graphs = open('all_graphs.txt', 'w')   #Stores all the graphs info
        self.threshold = 8                              #Minimum std deviation for storing a node
        self.edge_threshold = 50                        #Minimum distance for storing a line between nodes
        self.merge_thresh = 3                           #Similarity of std deviation for merging nodes
        self.filter_min = 0                             #Minimum value for thresholding filter
        self.filter_max = 255                           #Maximum value for thresholding filter
        self.jumper_i = 40                              #Number of pixels in i
        self.jumper_j = 40                              #Number of pixels in j
        self.img_height = 589                           #Size of the input image
        self.img_width = 1281                           #Size of the input image  
        
        #Control Variables
        self.bridge = CvBridge()                        #Bridge between cv and ros images
        self.jump_i = self.img_height/self.jumper_i     #Window size in i
        self.jump_j = self.img_width/self.jumper_j      #Window size in j
        
    #Returns a ROS image from a opencv image
    def convertImage(self, image):
        return self.bridge.cv2_to_imgmsg(image, "mono8")
        
    #Returns the original image and the initiliazed image
    def initImage(self, data):
        #creating lists
        self.centers_x = []
        self.centers_y = []
        self.std_x = []
        self.std_y = []
        
        #image to generate gaussians
        img = self.bridge.imgmsg_to_cv2(data, "mono8")
        
        #Thresholding the image and grayscaling
        ret, self.new_img = cv2.threshold(img, self.filter_min, self.filter_max,cv2.THRESH_BINARY)
        #self.new_img = cv2.cvtColor(self.new_img, cv2.COLOR_BGR2GRAY)
        
        #Setting up the mask
        mask = self.new_img.copy()    
        for i in range(0, 490):
            for j in range(173, 1174):
                mask[i][j] = 0   
             
        #Applying the mask
        self.new_img = self.new_img - mask
        return img, self.new_img
        
    #Returns gaussian image
    #Creates the segments and allocates the centers and standard deviations
    def createSegments(self, save_gaussian_image=False, gaussian_image_name=''):
        #Initializing values for the segmentation
        last_i = 0
        last_j = 0
        
        #Initializing blank image
        gaussian_image = np.zeros((self.jumper_i+1, self.jumper_j+1, 1), np.uint8)
        pos_i = 0
        pos_j = 0

        
        #Goes through segments and get values
        while last_j < self.img_width - (self.jump_j+1):
            while last_i < self.img_height - (self.jump_i+1):
                valuex = []
                valuey = []
                for i in range(last_i, last_i + self.jump_i):
                    for j in range(last_j, last_j + self.jump_j):
                        if self.new_img[i][j] != 0:
                            valuex.append(i)
                            valuey.append(j)
       
                if valuex != []:
                    self.std_x.append(np.std(valuex))
                else:
                    self.std_x.append(0)
                if valuey != []:
                    self.std_y.append(np.std(valuey))
                else:
                    self.std_y.append(0)
                        
                self.centers_x.append(last_i + 24)
                self.centers_y.append(last_j + 40)
                last_i += self.jump_i
                
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
        
        #Export variables
        sum_std_x = 0
        sum_std_y = 0
        n_gaussians = 0
        n_edges = 0
        
        
        #Save ellipses if they are good enough
        for i in range(0, len(self.centers_x)):
            if self.std_x[i] + self.std_y[i] > self.threshold:
            
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
                
        #Save lines by euclidian distance
        for i in range(0, len(ellipses)-1):
            for j in range(i+1, len(ellipses)):
                if ellipses[j] != -1 and ellipses[i] != -1:
                    if sqrt((ellipses[j][0] - ellipses[i][0])**2 + (ellipses[j][1] - ellipses[i][1])**2) < self.edge_threshold:
                        lines.append(((ellipses[i][0], ellipses[i][1]), (ellipses[j][0], ellipses[j][1]), (255,0,0)))
                        n_edges += 1
        
        #Draw ellipses
        for el in ellipses:
            if el != -1:
                cv2.ellipse(self.new_img, (el[0], el[1]), el[2], degrees(el[3]), 0, 360, (255,0,0), 2)
        
        #Draw lines
        for line in lines:
            cv2.line(self.new_img, line[0], line[1], (255,0,0))
        
        #Write to file
        if write:           
            self.all_graphs.write(str(sum_std_x)+' ')
            self.all_graphs.write(str(sum_std_y)+' ')
            self.all_graphs.write(str(n_gaussians)+' ')
            self.all_graphs.write(str(n_edges)+'\n')
        
        return self.new_img
                  
    
    #Plots and saves the images                
    def plotImages(self, img, save_plot, plot_name, save_graph=False, graph_name='', show_plot=False):
        plt.subplot(1,2,2)
        plt.title('Thresholded')
        plt.imshow(self.new_img)
        plt.subplot(1,2,1)
        plt.title('Original')
        plt.imshow(img)
        if save_plot:
            plt.savefig(plot_name)
        if save_graph:
            cv2.imwrite(graph_name, self.new_img)    
        if show_plot:
            plt.show()
                
    

    
    
