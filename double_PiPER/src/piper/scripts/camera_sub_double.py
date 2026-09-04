#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()

        self.frame_R = None
        self.frame_L = None

        self.sub_R = rospy.Subscriber('/followerR/camera', Image, self.callback_R)
        self.sub_L = rospy.Subscriber('/followerL/camera', Image, self.callback_L)

        # ← ここが重要
        rospy.Timer(rospy.Duration(0.03), self.timer_callback)

    def callback_R(self, msg):
        try:
            self.frame_R = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(e)

    def callback_L(self, msg):
        try:
            self.frame_L = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(e)

    def timer_callback(self, event):
        if self.frame_R is None or self.frame_L is None:
            return

        h = min(self.frame_R.shape[0], self.frame_L.shape[0])
        frame_R_resized = cv2.resize(self.frame_R, (int(self.frame_R.shape[1]*h/self.frame_R.shape[0]), h))
        frame_L_resized = cv2.resize(self.frame_L, (int(self.frame_L.shape[1]*h/self.frame_L.shape[0]), h))

        combined = cv2.hconcat([frame_L_resized, frame_R_resized])
        scale = 0.7  # 50%に縮小
        combined_resized = cv2.resize(combined, None, fx=scale, fy=scale)

        # cv2.imshow("Left | Right", combined)
        cv2.imshow("Left | Right", combined_resized)
        cv2.waitKey(1)


def main():
    rospy.init_node('camera_subscriber', anonymous=True)
    ImageSubscriber()
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()