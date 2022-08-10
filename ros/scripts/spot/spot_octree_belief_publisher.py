#!/usr/bin/env python
#
# Generates octree belief from given point cloud
import argparse
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2

class SpotOctreeBeliefPublisher:
    def __init__(self, args):
        point_cloud_topic = args.point_cloud_topic
        self._cloud_sub = rospy.Subscriber(point_cloud_topic, PointCloud2, self._cloud_cb)

    def _cloud_cb(self, msg):
        points = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        print(len(points))

    def run(self):
        rospy.spin()


def main():
    rospy.init_node("spot_octree_belief_publisher")
    parser = argparse.ArgumentParser("Octree belief publisher from point cloud")
    parser.add_argument("point_cloud_topic", type=str, help="input point cloud to convert")
    parser.add_argument("--viz-topic", type=str, help="topic to publish visualization of the octree belief",
                        default="octree_belief_visual")
    args, _ = parser.parse_known_args()

    bgen = SpotOctreeBeliefPublisher(args)
    bgen.run()


if __name__ == "__main__":
    main()
