#!/usr/bin/env python

from __future__ import print_function

# import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Twist

from std_msgs.msg import Empty, String

import sys, select, termios, tty

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    bound_box_pub = rospy.Publisher('bebop/bound_box', String, queue_size = 1)
    # pub = rospy.Publisher('bebop/cmd_vel', Twist, queue_size = 1)
    # pub2 = rospy.Publisher('bebop/takeoff', Empty, queue_size = 1)
    # pub3 = rospy.Publisher('bebop/land', Empty, queue_size = 1)
    # empty_msg = Empty()
    rospy.init_node('bound_box_node')


    try:
        # print(vels(speed,turn))
        while(1):
            res_str = String()
            res_str.data = "Hello World"
            bound_box_pub.publish(res_str)

    except Exception as e:
        print(e)