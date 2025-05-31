更新商品位置
方法一（不推荐）
在/home/ydr/workspace/rd_wa/src/yahboomcar_nav/param/文件夹下修改my_waypoint.yaml文件，其中pose即为为了找到该商品，机器人所需要移动到的位姿
 
方法二（推荐）
在工作环境下”/home/ydr/workspace/rd_wa/”，终端输入roslaunch yahboomcar_nav yahboomcar_navigation.launch open_rviz:=true map:=supermarket.yaml。然后rviz界面使用2D Nav Goal 来设置商品位置。具体代码在yahboomcar_nav包中的waypoint_generator.py中实现。
 

更新商品数据
可在yahboomcar_nav 中的infer.py中修改需要修改商品信息和商品识别模型

更新超市地图
终端输入
roslaunch yahboomcar_nav laser_bringup.launch
roslaunch yahboomcar_nav yahboomcar_map.launch map_type:=gmapping
roslaunch yahboomcar_ctrl yahboom_keyboard.launch
使用键盘控制机器人移动，遍历整个超市
然后终端输入roslaunch yahboomcar_nav map_saver.launch来保存地图roslaunch yahboomcar_nav laser_bringup.launch

实时参数调节，如costmap参数
终端输入rosrun rqt_reconfigure rqt_reconfigure

