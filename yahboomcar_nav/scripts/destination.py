#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

class DestinationPub:
    def __init__(self):
        
        # 定义发布器和订阅器
        self.pub_location = rospy.Publisher('product_location', Point, queue_size=100)
        self.sub_index = rospy.Subscriber("product_index", Float32, self.sub_index_callback, queue_size=100)
        
        # 商品位置信息
        self.product_location_list = [(1, 2), (3, 4), (5, 6)]
    
    def sub_index_callback(self, index_msg):
        product_index = int(index_msg.data)
        print(f"Received product index: {product_index}")
        rospy.loginfo(f"Received product index")
        rospy.loginfo(f"Received product index: {index_msg}")
        self.publish_product_location(product_index)
    
    def publish_product_location(self, product_index):
        if 0 <= product_index < len(self.product_location_list):
            product_location = Point()
            product_location.x = self.product_location_list[product_index][0]
            product_location.y = self.product_location_list[product_index][1]
            product_location.z = 0
            
            # 发布商品位置信息
            rospy.loginfo(f"Publishing location of {product_index}: x = {product_location.x}, y = {product_location.y}")
            self.pub_location.publish(product_location)
        else:
            rospy.logwarn(f"Invalid product index: {product_index}")

if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('product_location_publisher', anonymous=False)
    try:
        destination_pub = DestinationPub()
        rospy.spin()
    except:
        rospy.loginfo("Final!!!")
