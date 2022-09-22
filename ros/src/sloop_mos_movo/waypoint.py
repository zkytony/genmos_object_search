# note: code from MOS3D
# /author: Kaiyu Zheng
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moveit_msgs.msg import MoveItErrorCodes
import math
from scipy.spatial.transform import Rotation as scipyR
moveit_error_dict = {}
for name in MoveItErrorCodes.__dict__.keys():
    if not name[:1] == '_':
        code = MoveItErrorCodes.__dict__[name]
        moveit_error_dict[code] = name

def euclidean_dist(p1, p2):
    dist = math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))
    return dist

def yaw_diff(quat1, quat2):
    euler1 = scipyR.from_quat(quat1).as_euler("xyz")
    euler2 = scipyR.from_quat(quat2).as_euler("xyz")
    return abs(euler1[2] - euler2[2])

class WaypointApply(object):
    class Status:
        NOT_RUNNING = "not_running"
        RUNNING = "running"
        SUCCESS = "success"
        FAIL = "fail"
    def __init__(self,
                 position, orientation,
                 action_name="navigate",
                 xy_tolerance=0.1, rot_tolerance=0.3):
        # Get an action client
        self.client = actionlib.SimpleActionClient('movo_move_base', MoveBaseAction)
        rospy.loginfo("Waiting for movo_move_base AS...")
        if not self.client.wait_for_server(rospy.Duration(20)):
            rospy.logerr("Could not connect to movo_move_base AS")
            exit()
        rospy.loginfo("Connected!")
        rospy.sleep(1.0)

        self.status = WaypointApply.Status.NOT_RUNNING
        self.action_name = action_name
        self._position = position
        self._orientation = orientation
        self._xy_tolerance = xy_tolerance
        self._rot_tolerance = rot_tolerance
        self._goal_reached = False

        # Define the goal
        rospy.loginfo("Waypoint (%.2f,%.2f) and (%.2f,%.2f,%.2f,%.2f) is sent.", position[0], position[1], orientation[0], \
            orientation[1], orientation[2], orientation[3])
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = 0.0
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        self.waypoint_execute()

    def waypoint_execute(self):
        self.status = WaypointApply.Status.RUNNING
        self.client.send_goal(self.goal, self.done_cb, feedback_cb=self.feedback_cb)
        delay = rospy.Duration(0.1)
        while not self.client.wait_for_result(delay) and not rospy.is_shutdown():
            if self._goal_reached:
                rospy.loginfo("Goal has been reached by the robot actually. So cancel goal.")
                self.status = WaypointApply.Status.SUCCESS
                self.client.cancel_goal()
                break
            if self.status == WaypointApply.Status.FAIL:
                rospy.logerr("Could not reach goal.")
                self.client.cancel_goal()
                break

    def feedback_cb(self, feedback):
        base_position = feedback.base_position
        curx = base_position.pose.position.x
        cury = base_position.pose.position.y
        curz = base_position.pose.position.z
        curqx = base_position.pose.orientation.x
        curqy = base_position.pose.orientation.y
        curqz = base_position.pose.orientation.z
        curqw = base_position.pose.orientation.w
        # Check if already reached goal
        dist = euclidean_dist((curx, cury, curz), self._position)
        angle = yaw_diff((curqx, curqy, curqz, curqw), self._orientation)
        rospy.loginfo("(feedback)[dist_gap: %.5f     | angle_gap: %.5f]" % (dist, angle))
        if dist <= self._xy_tolerance\
           and angle <= self._rot_tolerance:
            self._goal_reached = True
            rospy.loginfo("Goal already reached within tolerance.")


    def done_cb(self, status, result):
        # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
        if status == 2:
            rospy.loginfo("Navigation action "+str(self.action_name)+" received a cancel request after it started executing, completed execution!")
            self.status = WaypointApply.Status.FAIL
        elif status == 3:
            rospy.loginfo("Navigation action "+str(self.action_name)+" reached")
            self.status = WaypointApply.Status.SUCCESS
        elif status == 4:
            rospy.loginfo("Navigation action "+str(self.action_name)+" was aborted by the Action Server")
            rospy.signal_shutdown("Navigation action "+str(self.action_name)+" aborted, shutting down!")
            self.status = WaypointApply.Status.FAIL
        elif status == 5:
            rospy.loginfo("Navigation action "+str(self.action_name)+" has been rejected by the Action Server")
            rospy.signal_shutdown("Navigation action "+str(self.action_name)+" rejected, shutting down!")
            self.status = WaypointApply.Status.FAIL
        elif status == 8:
            rospy.loginfo("Navigation action "+str(self.action_name)+" received a cancel request before it started executing, successfully cancelled!")
            self.status = WaypointApply.Status.FAIL
