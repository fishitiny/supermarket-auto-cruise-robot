import rospy
from geometry_msgs.msg import Point



sub_location = rospy.Subscriber('product_location', Point, sub_location_callback, queue_size=1)

        
if __name__ == '__main__':
    # 初始化节点
    rospy.init_node('location_receive', anonymous=False)
    try:
        rospy.spin()
    except:
        rospy.loginfo("Final!!!")