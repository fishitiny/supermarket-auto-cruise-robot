#!/usr/bin/env python3
# encoding: utf-8
import yaml
import rospy
import geometry_msgs.msg as geometry_msgs
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import move_base_msgs.msg as move_base_msgs
class WaypointGenerator(object):

    def __init__(self, filename):
        # 通过订阅rviz发出的/clicked_point的rostopic，就可以获取鼠标点击的目标点了：
        # Subscriber订阅/clicked_point的同时，交给作为callback函数的_process_pose去处理路标点：
        self._sub_pose = rospy.Subscriber('/move_base/goal', move_base_msgs.MoveBaseActionGoal
, self._process_pose, queue_size=1)
        self._waypoints = []
        self._filename = filename

    def _process_pose(self, msg):
        position = msg.goal.target_pose.pose.position
        orientation = msg.goal.target_pose.pose.orientation

        data = {}
        data['frame_id'] = msg.header.frame_id
        data['pose'] = {}
        # 因为rviz输出的是2D Nav Goal所以只处理2维：
        data['pose']['position'] = {'x': position.x, 'y': position.y, 'z': 0.0}
        data['pose']['orientation'] = {'x': 0, 'y': 0, 'z': orientation.z, 'w':orientation.w}
        data['name'] = '%s_%s' % (position.x, position.y)

        self._waypoints.append(data)
        rospy.loginfo("Clicked : (%s, %s, %s)" % (position.x, position.y, position.z))

    def _write_file(self):
        ways = {}
        ways['waypoints'] = self._waypoints
        # 把目标点输出成yaml文件：
        with open(self._filename, 'w') as f:
            f.write(yaml.dump(ways, default_flow_style=False))

    def spin(self):
        rospy.spin()
        self._write_file()


if __name__ == '__main__':

    rospy.init_node('waypoint_generator')
    filename = rospy.get_param('~filename')

    g = WaypointGenerator(filename)
    rospy.loginfo('Initialized')
    g.spin()
    rospy.loginfo('ByeBye')
