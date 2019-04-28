#!/usr/bin/env python

import roslib
import sys
import rospy
# wget download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
# tar -xzvf ssd_inception_v2_coco_2018_01_28.tar.gz
print("SYS V", sys.version)
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
import time
import os
#comment below 3 lines if no gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# pub = rospy.Publisher('/bbox', String, queue_size=10)
bound_box_pub = rospy.Publisher('bebop/bound_box', String, queue_size = 1)
# rospy.init_node('bound_box_node')

f_name = int(time.time())
os.system("mkdir videos/"+str(f_name))

# cap=cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter("video.avi", fourcc, 30.0, (480, 856))

class ImageConverter:

  def __init__(self):
    self.count = 0
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/bebop/image_raw",Image,self.callback)
    self.detectRate = 5
    self.threshold = 0.9

    #######DEFINE topic and publisher

  def callback(self,data):
    start_time = time.time()
    self.count = self.count+1
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
      return

    #cv2.imwrite(str(self.count)+'.png',cv_image)

    #threshold = 0.7
    if self.count % self.detectRate == 0:

        boxes, scores, classes, num = odapi.processFrame(cv_image)
        # Visualization of the results of a detection.
        # print(cv_image.shape)
        # print(boxes)
        # 240, 428
        i = classes.index(1)
        #score = scores[i]
        box = boxes[i]
        print((box[1]+box[3])*0.5, (box[0]+box[2])*0.5,)
        
        #######Publish here
        
        bound_box_pub.publish(",".join([str(box[1]),str(box[0]),str(box[3]),str(box[2])]))
        cv2.rectangle(cv_image, (box[1],box[0]),(box[3],box[2]), (255,0,0),2)


    # cv2.imwrite("videos/"+str(f_name)+"/"+str(self.count)+" "+str(time.time())+".png", cv_image)
    # n_image = cv2.flip(cv_image, 180)
    # out.write(cv_image)
    cv2.imshow("preview", cv_image)
    key = cv2.waitKey(1)
    end_time = time.time()
    print("Elapsed Time:", end_time-start_time)


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.first_run()

    def first_run(self):
        #tensorflow first predictions are ridiculously slow
        #our way tio hack around it
        test_img = np.zeros((1,480,856,3))
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: test_img})
        return


    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        print(image.shape)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()
        print("Model time", end_time-start_time)
        #print("Elapsed Time:", end_time-start_time)
        # print()
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

model_path = './src/image_subscriber/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
# model_path = './src/image_subscriber/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)

def main(args):
    ic = ImageConverter()
    rospy.init_node('image_converter', anonymous=True)
    rospy.spin()
  # except KeyboardInterrupt:
    print("Shutting down")
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    cv2.destroyAllWindows()
