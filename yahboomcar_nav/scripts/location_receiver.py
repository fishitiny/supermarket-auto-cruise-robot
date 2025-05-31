#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point

def sub_location_callback(location):
    # 处理接收到的商品位置信息
    rospy.loginfo(f"Received location: x = {location.x}, y = {location.y}, z = {location.z}")
    # 在这里可以添加其他处理逻辑，例如存储位置、发送到其他节点等

sub_location = rospy.Subscriber('product_location', Point, sub_location_callback, queue_size=1)


        
if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('location_receive', anonymous=False)
    try:
        rospy.spin()
    except:
        rospy.loginfo("Final!!!")
