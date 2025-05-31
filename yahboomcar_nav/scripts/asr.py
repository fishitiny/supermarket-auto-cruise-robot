#! /home/ydr/miniconda3/bin/python

import os,time
import  requests

import rospy
import sys

rospy.loginfo("sasa")
sys.path.append("")
print(sys.version)
login_url = "https://xha.ouc.edu.cn:802/eportal/portal/login"

print("hello")