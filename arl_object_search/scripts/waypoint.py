#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from arl_nav_msgs.msg import GotoRegionActionGoal 
from actionlib_msgs.msg import GoalStatusArray

class WaypointApply(object):
    class Status:
        NOT_RUNNING = "not_running"
        RUNNING = "running"
        SUCCESS = "success"
        FAIL = "fail"
    def __init__(self,
                 position, orientation,
                 action_name="navigate",
                 xy_tolerance=2.0, rot_tolerance=0):


        self.status = WaypointApply.Status.NOT_RUNNING

        self.action_name = action_name
        self._position = position
        self._orientation = orientation
        self._xy_tolerance = xy_tolerance
        self._rot_tolerance = rot_tolerance
        self._goal_reached = False

        rospy.init_node('talker', anonymous=True)
        # self.subscriber = rospy.Subscriber("warty/goto_region/status", GoalStatusArray, self.callback)
        #Define the publisher
        self.publisher = rospy.Publisher('warty/goto_region/goal', GotoRegionActionGoal, queue_size=10)
        

        # Define the goal
        rospy.loginfo("Waypoint (%.2f,%.2f) and (%.2f,%.2f,%.2f,%.2f) is sent.", position[0], position[1], orientation[0], \
            orientation[1], orientation[2], orientation[3])
        self.command = GotoRegionActionGoal()

        self.command.header.seq = 0
        self.command.header.stamp.secs = 0
        self.command.header.stamp.nsecs = 0
        self.command.header.frame_id = 'warty/map'

        self.command.goal_id.stamp.secs = 0
        self.command.goal_id.stamp.nsecs = 0
        self.command.goal_id.id = ''

        self.command.goal.region_center.header.seq = 0
        self.command.goal.region_center.header.stamp.secs = 0
        self.command.goal.region_center.header.stamp.nsecs = 0
        self.command.goal.region_center.header.frame_id = 'warty/map'

        self.command.goal.region_center.pose.position.x = position[0]
        self.command.goal.region_center.pose.position.y = position[1]
        self.command.goal.region_center.pose.position.z = position[2]

        self.command.goal.region_center.pose.orientation.x = orientation[0]
        self.command.goal.region_center.pose.orientation.y = orientation[1]
        self.command.goal.region_center.pose.orientation.z = orientation[2]
        self.command.goal.region_center.pose.orientation.w = orientation[3]

        self.command.goal.radius = xy_tolerance
        self.command.goal.angle_threshold = rot_tolerance
        self.command.goal.absolute_duration_limit = 0.0
        self.command.goal.relative_duration_limit = 0.0

        self.waypoint_execute()
        rospy.spin()
        

    def waypoint_execute(self):
        self.status = WaypointApply.Status.RUNNING
        counter = 0 
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            counter += 1
            if counter > 2:
                break
            rospy.loginfo(self.command)
            self.publisher.publish(self.command)
            rate.sleep()

if __name__ == '__main__':
    try:
        WaypointApply(position=(10.0, 10.0, 0), orientation=(0,0,0,1.0))
    except rospy.ROSInterruptException:
        pass


