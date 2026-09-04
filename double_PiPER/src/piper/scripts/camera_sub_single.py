#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/camera', Image, self.callback)
        # self.sub = rospy.Subscriber('/follower/camera', Image, self.callback)
        # self.sub = rospy.Subscriber('/followerR/camera', Image, self.callback)
        # self.sub = rospy.Subscriber('/followerL/camera', Image, self.callback)

    def callback(self, msg):
        try:
            # ROS → OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # print("frame shape:", frame.shape)
            cv2.imshow("Camera Subscriber", frame)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(e)

def main():
    rospy.init_node('camera_subscriber', anonymous=True)
    ImageSubscriber()
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
