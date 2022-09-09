#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from arl_nav_msgs.msg import GotoRegionActionGoal 


def talker():
    pub = rospy.Publisher('warty/goto_region/goal', GotoRegionActionGoal, queue_size=10)
    cmd = GotoRegionActionGoal()
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 10hz
    counter = 0 
    while not rospy.is_shutdown():
        counter += 1
        if counter > 2:
            break
        # cmd = GotoRegionActionGoal()
        cmd.header.seq = 0
        cmd.header.stamp.secs = 0
        cmd.header.stamp.nsecs = 0
        cmd.header.frame_id = 'warty/map'

        cmd.goal_id.stamp.secs = 0
        cmd.goal_id.stamp.nsecs = 0
        cmd.goal_id.id = ''

        cmd.goal.region_center.header.seq = 0
        cmd.goal.region_center.header.stamp.secs = 0
        cmd.goal.region_center.header.stamp.nsecs = 0
        cmd.goal.region_center.header.frame_id = 'warty/map'

        cmd.goal.region_center.pose.position.x = 10.0
        cmd.goal.region_center.pose.position.y = 10.0
        cmd.goal.region_center.pose.position.z = 0.0

        cmd.goal.region_center.pose.orientation.x = 0.0
        cmd.goal.region_center.pose.orientation.y = 0.0
        cmd.goal.region_center.pose.orientation.z = 0.0
        cmd.goal.region_center.pose.orientation.w = 1.0

        cmd.goal.radius = 1.0
        cmd.goal.angle_threshold = 0.0
        cmd.goal.absolute_duration_limit = 0.0
        cmd.goal.relative_duration_limit = 0.0

        # cmd.goal.boundary.points.x = 0.0
        # cmd.goal.boundary.points.y = 0.0
        # cmd.goal.boundary.points.z = 0.0
        # hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(cmd)
        pub.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass



"""
The rostopic publish this code is trying to replicate

"""

# rostopic pub -r 1 /warty/goto_region/goal arl_nav_msgs/GotoRegionActionGoal "
# header:
#   seq: 0
#   stamp:
#     secs: 0
#     nsecs: 0
#   frame_id: 'warty/map'
# goal_id:
#   stamp:
#     secs: 0
#     nsecs: 0
#   id: ''
# goal:
#   region_center:
#     header:
#       seq: 0
#       stamp: {secs: 0, nsecs: 0}
#       frame_id: 'warty/map'
#     pose:
#       position: {x: 10.0, y: 10.0, z: 0.0}
#       orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
#   radius: 2.0
#   angle_threshold: 0.0
#   absolute_duration_limit: 0.0
#   relative_duration_limit: 0.0
#   boundary:
#     points:
#     - {x: 0.0, y: 0.0, z: 0.0}"