#!/usr/bin/env python
import rospy
from extractor import GaussianFeatureExtractor
from sensor_msgs.msg import Image

class SonarNodes(object):

    #Initializes the node
    def __init__(self):
        rospy.init_node('sonar_node', anonymous=True)
        self.extract = GaussianFeatureExtractor()
    
    #Starts to listen to sonar images
    def sonar_listener(self):
        rospy.Subscriber("sonar_image", Image, self.callback)

    def gaussian_publisher(self):
        self.pub_gauss = rospy.Publisher('gaussian_image', Image, queue_size=10)
        
    def graph_publisher(self):
        self.pub_graph = rospy.Publisher('graph_image', Image, queue_size=10)
        
    def callback(self, data):
        img, new_img = self.extract.initImage(data)
        
        #publishing gaussian image
        gaussian_image = self.extract.createSegments()
        self.pub_gauss.publish(self.extract.convertImage(gaussian_image))
        
        #publishing graph image
        graph_image = self.extract.drawGraph()
        self.pub_graph.publish(self.extract.convertImage(graph_image))
        

if __name__ == '__main__':
    try:
        sonar_node = SonarNodes()
        sonar_node.gaussian_publisher()
        sonar_node.graph_publisher()
        sonar_node.sonar_listener()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass

