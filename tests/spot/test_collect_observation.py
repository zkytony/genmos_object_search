import rospy
from sloop_object_search.ros.ros_utils import WaitForMessages
from geometry_msgs.msg import PoseStamped
from rbd_spot_perception.msg import SimpleDetection3DArray

def test():
    rospy.init_node("test_collect_observation")
    messages = WaitForMessages(
        ["/spot_hand_pose"],
        [PoseStamped, SimpleDetection3DArray],
        delay=1.0,
        sleep=0.05,
        verbose=True).messages
    print("number of messages:", messages)
    print([type(m) for m in messages])


if __name__ == "__main__":
    test()
