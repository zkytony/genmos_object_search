import rospy
import time
import rbd_spot
from sloop_object_search.oopomdp.models.policy_model import PolicyModelBasic2D
from sloop_object_search_ros.msg import GridMap2d, KeyValAction
from sloop_object_search_ros.grid_map_utils import ros_msg_to_grid_map
from sloop_object_search.oopomdp.models.transition_model import RobotTransBasic2D
from sloop_object_search.utils.math import to_rad


def test_find_action(pub, wait=5):
    action_msg = KeyValAction(stamp=rospy.Time.now(),
                              type="find",
                              keys=[],
                              values=[])
    pub.publish(action_msg)
    time.sleep(wait)

def test():
    rospy.init_node("test_find_action")
    pub = rospy.Publisher("/run_sloop_mos_spot/action", KeyValAction, queue_size=10, latch=True)
    test_find_action(pub)


if __name__ == "__main__":
    test()
