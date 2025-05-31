#!/usr/bin/env python3
# encoding: utf-8
import sys
import math
import rospy
import random
import threading
from math import pi
from time import sleep
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, MagneticField, JointState
#from Rosmaster_Lib import Rosmaster
from robomaster import robot
from robomaster import chassis,blaster
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Int32, Bool
from dynamic_reconfigure.server import Server
from yahboomcar_bringup.cfg import PIDparamConfig
from std_msgs.msg import Float32MultiArray

class yahboomcar_driver:
    def __init__(self):
        rospy.on_shutdown(self.cancel)
       
        # Radians turn angle
        self.RA2DE = 180 / pi
        self.car = robot.Robot()
        self.car.initialize(conn_type="rndis")
        self.ep_blaster = self.car.blaster
        self.gimbal = self.car.gimbal
        self.car.chassis.sub_imu(freq=10, callback=self.imu_callback)
        self.car.chassis.sub_velocity(freq=10, callback=self.vel_callback)
        self.car.chassis.sub_position(freq=10, callback=self.position_callback)  
        self.car.chassis.sub_attitude(freq=10, callback=self.attitude_callback)  # Subscribe to attitude
        self.car.set_robot_mode(mode='chassis_lead')
        self.gimbal.recenter().wait_for_completed()
        self.is_first = False
        #time.sleep(3)
        self.search_flag = False
        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0
        self.gyro_x = 0
        self.gyro_y = 0
        self.gyro_z = 0
        
        self.Kp = 0.01  
        self.tolerance = 0.075  
        self.error_x = 0
        self.error_y = 0
        self.destination = -1
        self.gimbal_range=[[0,30],[0,60],[0,90],[-15,90],[-15,60],[-15,30],[-15,0],[-15,-30],[-15,-60],[-15,-90],[0,-90],[0,-60],[0,-30],[0,0],[0,30],[0,60],[0,90],[30,90],[30,60],[30,30],[30,0],[30,-30],[30,-60],[30,-90],[0,-90],[0,-60],[0,-30],[0,0]]
        self.current_range_index = 0
        self.body_vx = 0.0   # vbx
        self.body_vy = 0.0   # vby
        self.body_vz = 0.0   # vbz
        self.pos_x = 0.0     # x position
        self.pos_y = 0.0     # y position
        self.pos_z = 0.0     # z position
        self.yaw = 0.0       # yaw angle
        self.pitch = 0.0     # pitch angle
        self.roll = 0.0      # roll angle
        #self.car.set_car_type(1)
        self.imu_link = rospy.get_param("~imu_link", "imu_link")
        self.Prefix = rospy.get_param("~prefix", "")
        self.xlinear_limit = rospy.get_param('~xlinear_speed_limit', 1.0)
        self.ylinear_limit = rospy.get_param('~ylinear_speed_limit', 1.0)
        self.angular_limit = rospy.get_param('~angular_speed_limit', 5.0)
        self.sub_cmd_vel = rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        #self.sub_RGBLight = rospy.Subscriber("RGBLight", Int32, self.RGBLightcallback, queue_size=100)
        #self.sub_Buzzer = rospy.Subscriber("Buzzer", Bool, self.Buzzercallback, queue_size=100)
        self.EdiPublisher = rospy.Publisher('edition', Float32, queue_size=100)
        self.volPublisher = rospy.Publisher('voltage', Float32, queue_size=100)
        self.staPublisher = rospy.Publisher('joint_states', JointState, queue_size=100)
        self.velPublisher = rospy.Publisher("/pub_vel", Twist, queue_size=100)
        self.imuPublisher = rospy.Publisher("/pub_imu", Imu, queue_size=100)
        self.magPublisher = rospy.Publisher("/pub_mag", MagneticField, queue_size=100)
        self.modePublisher = rospy.Publisher("/stop_mode", Bool, queue_size=1)
        self.odoPublisher = rospy.Publisher("/odom", Odometry, queue_size=100)
        self.sub_index = rospy.Subscriber("destination_index", Float32, self.sub_index_callback, queue_size=1)
        self.sub_location = rospy.Subscriber('label_location', Float32MultiArray,self.sub_labelLocation_callback, queue_size=100)
        self.stopPublisher = rospy.Publisher('stop_mode', Bool, queue_size=1)
        self.sub_mode1 = rospy.Subscriber('mode', Int32,self.mode1_callback, queue_size=1)
        self.dyn_server = Server(PIDparamConfig, self.dynamic_reconfigure_callback)

        #self.car.create_receive_threading()
        
        
    def pid_control(self,error, Kp):
        # integral += error
        # derivative = error - previous_error
        control = Kp * error 
        return control
        
    def cancel(self):
        self.gimbal.recenter().wait_for_completed()
        self.velPublisher.unregister()
        self.imuPublisher.unregister()
        self.EdiPublisher.unregister()
        self.volPublisher.unregister()
        self.staPublisher.unregister()
        self.magPublisher.unregister()
        self.odoPublisher.unregister()
        self.sub_cmd_vel.unregister()
        self.modePublisher.unregister()
        self.sub_index.unregister()
        self.sub_location.unregister()
        self.stopPublisher.unregister()
        #self.sub_RGBLight.unregister()
        #self.sub_Buzzer.unregister()
        self.car.close()
        # Always stop the robot when shutting down the node
        rospy.loginfo("Close the robot...")
        rospy.sleep(1)

    def pub_data(self):
       
        ## Publish the speed of the car, gyroscope data, and battery voltage
        while not rospy.is_shutdown():
            #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            sleep(0.05)
            imu = Imu()
            odo = Odometry()
            twist = Twist()
            battery = Float32()
            edition = Float32()
            mag = MagneticField()
            state = JointState()
            state.header.stamp = rospy.Time.now()
            state.header.frame_id = "joint_states"
            if len(self.Prefix)==0:
                state.name = ["front_right_joint", "front_left_joint",
                              "back_right_joint", "back_left_joint"]
            else:
                state.name = [self.Prefix+"/front_right_joint", self.Prefix+"/front_left_joint",
                              self.Prefix+"/back_right_joint", self.Prefix+"/back_left_joint"]
            #edition.data = self.car.get_version()
            #battery.data = self.car.get_battery_voltage()
            #ax, ay, az = self.car.get_accelerometer_data()
            ax, ay, az, gx, gy, gz = self.get_accelerate_data()
            #gx, gy, gz = self.car.get_gyroscope_data()
            #mx, my, mz = self.car.get_magnetometer_data()
            vx, vy, angular = self.get_vel_data()
            
            pos_x, pos_y, pos_z, yaw, pitch, roll = self.get_odometry_data()
            # 发布陀螺仪的数据
            # Publish gyroscope data
            imu.header.stamp = rospy.Time.now()
            imu.header.frame_id = self.imu_link
            imu.linear_acceleration.x = -ay
            imu.linear_acceleration.y = -ax
            imu.linear_acceleration.z = -az
            imu.angular_velocity.x =  gx/self.RA2DE
            imu.angular_velocity.y =  gy/self.RA2DE
            imu.angular_velocity.z = -gz/self.RA2DE
            
            odo.header.stamp = rospy.Time.now()
            odo.header.frame_id = "odom"
            odo.pose.pose.position.x = -pos_y
            odo.pose.pose.position.y = -pos_x
            odo.pose.pose.position.z = -pos_z
            odo.twist.twist.angular.x = -yaw/self.RA2DE
            odo.twist.twist.angular.y = 0
            odo.twist.twist.angular.z = 0
            
            mag.header.stamp = rospy.Time.now()
            mag.header.frame_id = self.imu_link
            mag.magnetic_field.x = 0
            mag.magnetic_field.y = 0
            mag.magnetic_field.z = 0
           
            # Publish the current linear vel and angular vel of the car
            twist.linear.x = -vy
            twist.linear.y = -vx
            twist.angular.z = -angular/self.RA2DE
            self.velPublisher.publish(twist)
            self.odoPublisher.publish(odo)
            # print("ax: %.5f, ay: %.5f, az: %.5f" % (ax, ay, az))
            # print("gx: %.5f, gy: %.5f, gz: %.5f" % (gx, gy, gz))
            # print("mx: %.5f, my: %.5f, mz: %.5f" % (mx, my, mz))
            # rospy.loginfo("battery: {}".format(battery))
            # rospy.loginfo("vx: {}, vy: {}, angular: {}".format(twist.linear.x, twist.linear.y, twist.angular.z))
            self.imuPublisher.publish(imu)
            #self.magPublisher.publish(mag)
            #self.volPublisher.publish(battery)
            #self.EdiPublisher.publish(edition)
            state.position = [0, 0, 0, 0]
            if not vx == vy == angular == 0:
                i = random.uniform(-3.14, 3.14)
                state.position = [i, i, i, i]
            #self.staPublisher.publish(state)

    def position_callback(self, data):
        self.pos_x, self.pos_y, self.pos_z = data

    def attitude_callback(self, data):
        self.yaw, self.pitch, self.roll = data
        
    def get_odometry_data(self):
        
        return self.pos_x, self.pos_y, self.pos_z, self.yaw, self.pitch, self.roll
        
    def imu_callback(self, data):
       
        # print(data)
        # print(type(data))
        self.acc_x, self.acc_y,self.acc_z,self.gyro_x,self.gyro_y,self.gyro_z =data
        # print(self.acc_x)


    def get_accelerate_data(self):
       
        return self.acc_x, self.acc_y, self.acc_z,self.gyro_x,self.gyro_y,self.gyro_z

    def vel_callback(self, data):
        #print(data)
        self.body_vx ,self.body_vy,self.body_vz,a,b,c = data
    def get_vel_data(self):
       
        return self.body_vx,self.body_vy,self.body_vz

   

    def cmd_vel_callback(self, msg):
      
        # Car motion control, subscriber callback function
        if not isinstance(msg, Twist): return
      
        # Issue linear vel and angular vel
        vx = msg.linear.x
        vy = msg.linear.y
        angular = msg.angular.z
       
        # Trolley motion control,vel=[-1, 1], angular=[-5, 5]
        # rospy.loginfo("cmd_velx: {}, cmd_vely: {}, cmd_ang: {}".format(vx, vy, angular))
        #self.car.set_car_motion(vx, vy, angular)

        self.car.chassis.drive_speed(x=-vy, y=-vx, z=-angular*self.RA2DE, timeout=5)
        #self.car.chassis.drive_speed(x=0, y=0, z=0, timeout=5)
        #self.car.chassis.drive_speed(0, 0, 0, timeout=5)
        

    def dynamic_reconfigure_callback(self, config, level):
        # self.car.set_pid_param(config['Kp'], config['Ki'], config['Kd'])
        # print("PID: ", config['Kp'], config['Ki'], config['Kd'])
        self.linear_max = config['linear_max']
        self.linear_min = config['linear_min']
        self.angular_max = config['angular_max']
        self.angular_min = config['angular_min']
        return config
    
    def mode1_callback(self,msg):
      if msg.data == 1:
        self.gimbal.recenter().wait_for_completed()


    def sub_labelLocation_callback(self,location): 
      print(self.is_first)
      if self.is_first:
            if(location.data[0]==-1 and location.data[1]==-1 and location.data[2]==-1 and location.data[3]==-1 and location.data[4]==-1 and self.search_flag == False   ):
          
              print(self.gimbal_range[self.current_range_index][0],self.gimbal_range[self.current_range_index][1])
    
              self.gimbal.moveto(self.gimbal_range[self.current_range_index][0],self.gimbal_range[self.current_range_index][1]).wait_for_completed()
              self.current_range_index = self.current_range_index +1
              
              if(self.current_range_index >= len(self.gimbal_range)):
                  self.current_range_index = 0
    
            else:
              self.search_flag = True  
              error_x = location.data[0] - (location.data[3] / 2)
              error_y = location.data[1] - (location.data[2] / 2)
    
            #if location.data[4] == self.destination:
              if abs(error_x) < (self.tolerance * location.data[3]) and abs(error_y) < (self.tolerance * location.data[2]):
                  print("shoot")
                  self.gimbal.drive_speed(0,0)
                  self.ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=3)
                  stop = Bool()
                  stop.data = True
                  self.stopPublisher.publish(stop)
                  self.search_flag = False 
                  time.sleep(2) 
                  #self.gimbal.recenter().wait_for_completed()
                  self.current_range_index = 0
                  rospy.loginfo("Laser fired!")
                  self.is_first = False
              else:
    
                  control_x = self.pid_control(error_x, self.Kp)
                  control_y = self.pid_control(error_y, self.Kp)
    
                  self.gimbal.drive_speed(-control_y,control_x)
                  #self.time.sleep(0.2)
                
              rospy.loginfo(f"Received location: x = {location.x}, y = {location.y}, z = {location.z}")
      else:
                  # self.gimbal.recenter().wait_for_completed()
                  print("shoot")
                  self.gimbal.drive_speed(0,0)
                  self.ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=3)
                  stop = Bool()
                  stop.data = True
                  self.stopPublisher.publish(stop)
                  self.search_flag = False 
                  time.sleep(2) 
                  self.gimbal.recenter().wait_for_completed()
                  self.current_range_index = 0
                  rospy.loginfo("Laser fired!")
                  self.is_first = False
          

    def sub_index_callback(self,dest):
        self.destination = dest.data

if __name__ == '__main__':
    rospy.init_node("driver_node", anonymous=False)
    # try:
    driver = yahboomcar_driver()
    driver.pub_data()
    rospy.spin()
    # except:
    #     rospy.loginfo("Final!!!")
