#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('camera_publisher', anonymous=True)
    video_port = rospy.get_param("~video_port", 0)
    pub = rospy.Publisher('camera', Image, queue_size=10)

    bridge = CvBridge()
    cap = cv2.VideoCapture(int(video_port))  # 0 = default webcam

    rate = rospy.Rate(30)  # 30Hz

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (320, 240))
        if not ret:
            rospy.logwarn("Failed to capture image")
            continue

        # OpenCV → ROS Image message
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        pub.publish(ros_image)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    main()