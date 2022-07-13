import rospy
from sloop_object_search.ros.ros_utils import WaitForMessages
from geometry_msgs.msg import PoseStamped
from rbd_spot_perception.msg import SimpleDetection3DArray

def test():
    rospy.init_node("test_collect_observation")
    messages = WaitForMessages(
        ["/spot_hand_pose", "/spot/segmentation/hand/result_boxes3d"],
        [PoseStamped, SimpleDetection3DArray],
        delay=1.0,
        sleep=0.1).messages
    print("number of messages:", messages)
    print(len(messages))
    print(type(messages[0]))
    assert type(messages[0]) == PoseStamped
    print(type(messages[1]))
    assert type(messages[1]) == SimpleDetection3DArray


if __name__ == "__main__":
    test()
