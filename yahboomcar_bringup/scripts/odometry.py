#!/usr/bin/env python
# encoding: utf-8
import sys
import math
import rospy
import random
import threading
from math import pi
from time import sleep
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, MagneticField, JointState
#from Rosmaster_Lib import Rosmaster
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Int32, Bool
from dynamic_reconfigure.server import Server

import tf  
'''
# 初始化节点
rospy.init_node('odo_publisher')

sub_cmd_vel = rospy.Subscriber('odom', Odometry, self.cmd_vel_callback, queue_size=10)
pos_x = 0.0     # x position
pos_y = 0.0     # y position
pos_z = 0.0     # z position
yaw = 0.0       # yaw angle
pitch = 0.0     # pitch angle
roll = 0.0      # roll angle

# 创建一个静态变换广播器
static_broadcaster = tf2_ros.StaticTransformBroadcaster()

# 创建一个静态变换消息
static_transformStamped = TransformStamped()

# 设置变换的头部信息
static_transformStamped.header.stamp = rospy.Time.now()
static_transformStamped.header.frame_id = "odom"  # 父坐标系
static_transformStamped.child_frame_id = "base_footprint"  # 子坐标系

# 设置变换的平移部分
static_transformStamped.transform.translation.x = self.pos_x
static_transformStamped.transform.translation.y = self.pos_y
static_transformStamped.transform.translation.z = self.pos_z

# 设置变换的旋转部分（使用四元数）
static_transformStamped.transform.rotation.x = quaternion[0]
static_transformStamped.transform.rotation.y = quaternion[1]
static_transformStamped.transform.rotation.z = quaternion[2]
static_transformStamped.transform.rotation.w = quaternion[3]

# 广播静态变换
static_broadcaster.sendTransform(static_transformStamped)

# 保持节点运行
rospy.spin()

def cmd_vel_callback(self, msg):
      
        # Car motion control, subscriber callback function
        if not isinstance(msg, Odometry): return
        global pos_x = msg.pose.pose.position.x
        global pos_y = msg.pose.pose.position.y
        global pos_z =msg.pose.pose.position.z
        global = msg.twist.twist.angular.x
        global = msg.twist.twist.angular.y
        global = msg.twist.twist.angular.z

'''



class my_odometry:
    def __init__(self):
        rospy.on_shutdown(self.cancel)
       
        # Radians turn angle
        self.RA2DE = 180 / pi
        self.sub_cmd_vel = rospy.Subscriber('odom', Odometry, self.cmd_vel_callback, queue_size=10)

        self.pos_x = 0.0     # x position
        self.pos_y = 0.0     # y position
        self.pos_z = 0.0     # z position
        self.yaw = 0.0       # yaw angle
        self.pitch = 0.0     # pitch angle
        self.roll = 0.0      # roll angle
        #self.car.set_car_type(1)

        self.tf_broadcaster =  tf.TransformBroadcaster()  # Initialize tf broadcaster
        
    def cancel(self):
        self.sub_cmd_vel.unregister()
        # Always stop the robot when shutting down the node
        rospy.sleep(1)

    def pub_data(self):
       
        ## Publish the speed of the car, gyroscope data, and battery voltage
        while not rospy.is_shutdown():
            #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            sleep(0.05)
           
            self.publish_position_to_tf()

    def position_callback(self, data):
        self.pos_x, self.pos_y, self.pos_z = data

    def attitude_callback(self, data):
        self.yaw, self.pitch, self.roll = data

    def publish_position_to_tf(self):
        # 使用tf.TransformBroadcaster广播位置
        quaternion = tf.transformations.quaternion_from_euler(self.roll, self.pitch, self.yaw)
        self.tf_broadcaster.sendTransform(
            (self.pos_x, self.pos_y, self.pos_z),
            quaternion, 
            rospy.Time.now(),
            "base_footprint",  # Child frame
            "odom"        # Parent frame
        )
   
    def cmd_vel_callback(self, msg):
      
        # Car motion control, subscriber callback function
        if not isinstance(msg, Odometry): return
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z =msg.pose.pose.position.z
        self.yaw = msg.twist.twist.angular.x
        self.pitch = msg.twist.twist.angular.y
        self.roll = msg.twist.twist.angular.z


if __name__ == '__main__':
    rospy.init_node("odo_publisher", anonymous=False)
    # try:
    driver = my_odometry()
    driver.pub_data()
    rospy.spin()
    # except:
    #     rospy.loginfo("Final!!!")
    
    
