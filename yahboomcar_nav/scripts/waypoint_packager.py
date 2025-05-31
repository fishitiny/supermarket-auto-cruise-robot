#!/usr/bin/env python3
# encoding: utf-8
import random
import rospy
import actionlib # 引用actionlib库
import move_base_msgs.msg as move_base_msgs
import visualization_msgs.msg as viz_msgs
from std_msgs.msg import Float32,Int32
import yaml
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

id_count = 1

def get_waypoints(filename):
    # 目标点文件是yaml格式的：
    with open(filename, 'r') as f:
        data = yaml.load(f)

    return data['waypoints']

def create_geo_pose(p):
    pose = geometry_msgs.Pose()

    pose.position.x = p['pose']['position']['x']
    pose.position.y = p['pose']['position']['y']
    pose.position.z = p['pose']['position']['z']
    pose.orientation.x = p['pose']['orientation']['x']
    pose.orientation.y = p['pose']['orientation']['y']
    pose.orientation.z = p['pose']['orientation']['z']
    pose.orientation.w = p['pose']['orientation']['w']
    return pose

def create_move_base_goal(p):
    
    target = geometry_msgs.PoseStamped()
    target.header.frame_id = p['frame_id']
    target.header.stamp = rospy.Time.now()
    target.pose = create_geo_pose(p)
    
    goal = move_base_msgs.MoveBaseGoal(target), target
    return goal

def create_viz_markers(waypoints):
    marray= viz_msgs.MarkerArray()
    for w in waypoints:
        m_arrow = create_arrow(w)
        m_text = create_text(w)
        marray.markers.append(m_arrow)
        marray.markers.append(m_text)
    return marray

def create_marker(w):
    global id_count
    m = viz_msgs.Marker()
    m.header.frame_id = w['frame_id']
    m.ns = w['name']
    m.id = id_count
    m.action = viz_msgs.Marker.ADD
    m.pose = create_geo_pose(w)
    m.scale = geometry_msgs.Vector3(1.0,0.3,0.3)
    m.color = std_msgs.ColorRGBA(0.0,1.0,0.0,1.0)

    id_count = id_count + 1
    return m

def create_arrow(w):
    m = create_marker(w)
    m.type = viz_msgs.Marker.ARROW
    m.color = std_msgs.ColorRGBA(0.0,1.0,0.0,1.0)
    return m

def create_text(w):
    m = create_marker(w)
    m.type = viz_msgs.Marker.TEXT_VIEW_FACING
    m.pose.position.z = 2.5
    m.text = w['name']
    return m

class TourMachine(object):

    def __init__(self, filename, random_visits=False, repeat=False):
        self._waypoints = get_waypoints(filename) # 获取一系列目标点的值

        action_name = 'move_base'
        self._ac_move_base = actionlib.SimpleActionClient(action_name, move_base_msgs.MoveBaseAction) # 创建一个SimpleActionClient
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', geometry_msgs.PoseStamped, queue_size=10)
        self.mode_pub = rospy.Publisher('/mode', Int32, queue_size=10)
        self.sub_id = rospy.Subscriber('destination_index', Float32, self.move_to_next, queue_size=1)
        self.sub_result = rospy.Subscriber('move_base/status',GoalStatusArray, self.pub_mode, queue_size=100)

        rospy.loginfo('Wait for %s server' % action_name)
        self._ac_move_base.wait_for_server
        # self._counter = 0
        self._repeat = repeat
        self._random_visits = random_visits
        self.search_flag = False
        if self._random_visits:
            random.shuffle(self._waypoints)
        # 以下为了显示目标点：
        self._pub_viz_marker = rospy.Publisher('viz_waypoints', viz_msgs.MarkerArray, queue_size=1, latch=True)
        self._viz_markers = create_viz_markers(self._waypoints)

    def pub_mode(self,result):
        print(self.search_flag == True)
        #rospy.loginfo("Result : %b" % self.search_flag)
        if self.search_flag == True:
            print(type(result.status_list[-1].status))
        if(self.search_flag == True) and  (result.status_list[-1].status == 3) :
            self.search_flag = False
            mode = Int32()
            mode.data = 1
            self.mode_pub.publish(mode)
        
    def move_to_next(self,id):
        print(id)
        id = int(id.data)
        p = self._get_next_destination(id)
        
        if not p:
            rospy.loginfo("Finishing Tour")
            return True
        # 把文件读取的目标点信息转换成move_base的goal的格式：

        goal, simple_goal = create_move_base_goal(p)
        rospy.loginfo("Move to %s" % p['name'])
        # 这里也是一句很简单的send_goal:
        self._ac_move_base.send_goal(goal)
        self.goal_pub.publish(simple_goal)
        self._ac_move_base.wait_for_result()
        result = self._ac_move_base.get_result()
        rospy.loginfo("Result : %s" % result)
        self.search_flag = True
        return False

    def _get_next_destination(self,id):
        """
        根据是否循环，是否随机访问等判断，决定下一个目标点是哪个
        """
        if id >= len(self._waypoints):
            return self._waypoints[0]
        next_destination = self._waypoints[id]
        return next_destination

if __name__ == '__main__':
    rospy.init_node('tour')
    
    # 使用了ros的get_param读取文件名：
    filename = rospy.get_param('~filename')
    random_visits = rospy.get_param('~random', False)
    repeat = rospy.get_param('~repeat', False)
    '''
    #pub = rospy.Publisher()
    filename = "/home/ydr/workspace/rd_wa/src/yahboomcar_nav/param/my_waypoint.yaml"
    random_visits = False
    repeat = False
    '''
    m = TourMachine(filename, random_visits, repeat)
    rospy.loginfo('Initialized')
    rospy.spin()
    rospy.loginfo('Bye Bye')
