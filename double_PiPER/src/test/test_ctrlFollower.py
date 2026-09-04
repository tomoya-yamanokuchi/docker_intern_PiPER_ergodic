#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import rospy
from sensor_msgs.msg import JointState
from piper_sdk import Piper


class PiperFollower:

    def __init__(self):

        rospy.init_node("piper_follower_follower")

        self.move_spd_rate_ctrl = 40

        # ===== Piper init =====
        self.piper = Piper("can0")
        self.interface = self.piper.init()

        self.piper.connect()
        time.sleep(0.2)

        while not self.piper.enable_arm():
            time.sleep(0.01)

        self.piper.enable_gripper()

        self.interface.ModeCtrl(0x01, 0x01, self.move_spd_rate_ctrl, 0x00)

        print("Follower ready. Waiting /joint_grip_pose ...")

        # ===== Subscriber =====
        rospy.Subscriber(
            "/joint_grip_pose",
            JointState,
            self.joint_callback,
            queue_size=1
        )

    # =========================================================
    #  Leader → Follower 追従コア
    # =========================================================
    def joint_callback(self, msg):

        # msg.position:
        # [j1,j2,j3,j4,j5,j6,gripper]

        if len(msg.position) < 7:
            return

        joints = list(msg.position[:6])
        gripper = msg.position[6]

        # 念のため safety clamp
        gripper = max(0.0, min(0.1, gripper))

        # そのままSDKへ（変換不要！）
        self.piper.move_j(joints, self.move_spd_rate_ctrl)
        self.piper.move_gripper(gripper, 1)


# =============================================================
if __name__ == "__main__":
    node = PiperFollower()
    rospy.spin()
