#!/usr/bin/env python
# import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, String

pub = rospy.Publisher('bebop/cmd_vel', Twist, queue_size = 1)

def callback(data):
    # print(data.data)
    coordinates = data.data.split(',')
    for i in range(len(coordinates)):
        coordinates[i] = int(coordinates[i])

    #actual center = 240, 428
    area = (coordinates[0]-coordinates[2]) * (coordinates[1] - coordinates[3])
    center = (coordinates[0]+coordinates[2])/2, (coordinates[1]+coordinates[3])/2
    # print(coordinates, center, area)
    
    x,y,z,td = 0,0,0,0


    goback=0
    goclose=0
    if area>60000:
        goback=1
        x = -1
    elif area<30000:
        goclose=1
        x = 1
    if center[0]<428:
        td = 1
    elif center[0]>428:
        td = -1

    print(goback,goclose,td)

    ####
    #Send decision to drone
    ####

    speed=0.1
    turn=0.1

    twist = Twist()
    twist.linear.x = x*speed; twist.linear.y = y*speed; twist.linear.z = z*speed;
    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = td*turn
    pub.publish(twist)



if __name__=="__main__":

    # pub = rospy.Publisher('bebop/cmd_vel', Twist, queue_size = 1)
    # pub2 = rospy.Publisher('bebop/takeoff', Empty, queue_size = 1)
    # pub3 = rospy.Publisher('bebop/land', Empty, queue_size = 1)
    # empty_msg = Empty()
    rospy.init_node('listener')

    rospy.Subscriber('bebop/bound_box', String, callback)

    rospy.spin()