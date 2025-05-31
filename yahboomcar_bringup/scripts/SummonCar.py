#!/usr/bin/env python
# encoding: utf-8
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Int32, Bool
from sensor_msgs.msg import LaserScan
from time import sleep
from math import radians, copysign
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
import tf
from time import time
DEG2RAD = 180 * math.pi
from transform_utils import quat_to_angle, normalize_angle
from time import sleep
import numpy as np
import os
RAD2DEG = 180 / math.pi
class Summon_Go:
    def __init__(self):
        self.sub_flag = rospy.Subscriber("/pub_flag",Int32,self.excuteCallback,queue_size=1)
        self.sub_laser = rospy.Subscriber('/scan', LaserScan, self.registerScan, queue_size=1)
        self.cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.sub_angle = rospy.Subscriber("/sunmon_angel",Int32,self.getangleCallback,queue_size=1)
        self.obstacles = False
        self.vel = Twist()
        self.getangle = 0
        self.LaserAngle = 30
        self.ResponseDist = 0.55
        self.warning = 1
        self.rate = rospy.get_param('~rate', 20)
        self.r = rospy.Rate(self.rate) 
        
        #TF
        self.odom_frame = rospy.get_param('~odom_frame', '/odom')
        self.base_frame = rospy.get_param('~base_frame', '/base_footprint')
        self.tf_listener = tf.TransformListener()
        rospy.sleep(2)
        self.tf_listener.waitForTransform(self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(60.0))  
        #point 
        self.test_angle = 0
        self.tolerance = radians(rospy.get_param('tolerance', 1.5))
        self.reverse = 1
        self.done = False
        self.speed = 1.0
        self.odom_angular_scale_correction=rospy.get_param('~odom_angular_scale_correction', 1.0)
    def excuteCallback(self,msg):
        if msg.data == 1:
            self.vel.linear.x = 0.0
            self.vel.linear.y = 0.0
            #self.vel.angular.z = 0.3
            print("awakeup_angle:", self.getangle)
            self.vel.angular.z = 1
            if self.getangle<180 or self.getangle==180:
                self.getangle = self.getangle
            else:
                self.getangle = self.getangle
            print("self.getangle: ",self.getangle)
            self.adjust(self.getangle,self.vel.angular.z)
            print("finish")
            self.done = False
            sleep(3)
            '''self.vel.linear.x = 0.3
            self.vel.angular.z = 0.0
            self.cmd_vel.publish(self.vel)'''
            #sleep(3)
            #self.cmd_vel.publish(Twist())
           # os.system("rosrun yahboomcar_laser laser_Tracker.py")
            while self.warning<10:
                self.vel.linear.x = 0.3
                self.vel.angular.z = 0.0
                self.cmd_vel.publish(self.vel)
            self.cmd_vel.publish(Twist())
            
            '''rads = self.getangle/DEG2RAD
            adjust_time = abs(rads / self.vel.angular.z)*10
            print("adjust_time: ",abs(adjust_time))
            self.velPublisher.publish(self.vel)
            sleep(adjust_time)
            self.vel.angular.z = 0.0
            self.velPublisher.publish(self.vel)
            return'''
            '''while self.warning < 10:
                self.velPublisher.publish(self.vel)'''
            
    def getangleCallback(self,msg):
        self.getangle = msg.data
        
    def registerScan(self, scan_data):
        self.warning = 1
        if not isinstance(scan_data, LaserScan): return
        #if self.ros_ctrl.Joy_active == True: return
        # 记录激光扫描并发布最近物体的位置（或指向某点）
        ranges = np.array(scan_data.ranges)
        # 按距离排序以检查从较近的点到较远的点是否是真实的东西
        # if we already have a last scan to compare to:
        for i in range(len(ranges)):
            angle = (scan_data.angle_min + scan_data.angle_increment * i) * RAD2DEG
            # if angle > 90: print "i: {},angle: {},dist: {}".format(i, angle, scan_data.ranges[i])
            # 通过清除不需要的扇区的数据来保留有效的数据
            if abs(angle) > (180 - self.LaserAngle):
                if ranges[i] < self.ResponseDist: self.warning += 1
    
        
    def get_odom_angle(self):
        # Get the current transform between the odom and base frames
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        # Convert the rotation from a quaternion to an Euler angle
        return quat_to_angle(Quaternion(*rot))

            
    def adjust(self,adjust_angle,speed):
        print("111111")
        if adjust_angle<180 or adjust_angle>300:
            self.reverse = -self.reverse
            if adjust_angle>300:
                adjust_angle = 360 - adjust_angle
        else:
            self.reverse = self.reverse
            adjust_angle = 300 - adjust_angle
        adjust_angle = adjust_angle + 30
        while self.done == False:
            print("--------------------")

            self.odom_angle = self.get_odom_angle()
            last_angle = self.odom_angle
            turn_angle = 0
            adjust_angle = radians(adjust_angle)
            adjust_angle *= self.reverse
            error = adjust_angle - turn_angle
            self.reverse = self.reverse
            
            
            rospy.sleep(0.5)
            while abs(error) > self.tolerance:
                print("+++++++++++++++++++")
                start = time()
                move_cmd = Twist()
                move_cmd.angular.z = copysign(speed, error)
                self.cmd_vel.publish(move_cmd)
                self.r.sleep()
                self.odom_angle = self.get_odom_angle()
                delta_angle = self.odom_angular_scale_correction * normalize_angle(self.odom_angle - last_angle)
                turn_angle += delta_angle
                error = adjust_angle - turn_angle
                last_angle = self.odom_angle
                end = time()
                self.done = True
                rospy.loginfo(
                        "time: {},test_angle: {},turn_angle: {}".format((start - end), adjust_angle, turn_angle))
            self.cmd_vel.publish(Twist())   
    
        

if __name__ == '__main__':
    rospy.init_node("SummonCar_node",anonymous=False)
    sunmon_go = Summon_Go()
    rospy.spin()
        
