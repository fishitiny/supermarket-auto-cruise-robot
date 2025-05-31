#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

index = 0
pub_index = rospy.Publisher("product_index", Float32, queue_size=100)


if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('index_publisher', anonymous=False)
    try:
        pub_index.publish(index)
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
          pub_index.publish(index)
          rospy.loginfo(f"Published index: {index}")
          rate.sleep()
    except:
        rospy.loginfo("Final!!!")
        