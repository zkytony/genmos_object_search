#!/usr/bin/env python
#
# Generates octree belief from given point cloud
import argparse
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sloop_object_search.oopomdp.models.octree_belief import Octree, OctreeBelief
from sloop_mos_ros.conversion import convert, Frame
from sloop_object_search.oopomdp import ObjectState

OBJID = "obj"

VAL = 100

class SpotOctreeBeliefPublisher:
    def __init__(self, args):
        point_cloud_topic = args.point_cloud_topic
        self._cloud_sub = rospy.Subscriber(point_cloud_topic, PointCloud2, self._cloud_cb)
        octree = Octree(OBJID, (args.size, args.size, args.size))
        self._octree_belief = OctreeBelief(args.size, args.size, args.size,
                                           OBJID, "OBJ", octree)
        self._search_space_res = args.res


    def _cloud_cb(self, pcl2msg):
        pc = ros_numpy.numpify(pcl2msg)
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        origin = np.min(points, axis=0)
        for p in points:
            pomdp_point = convert(p, Frame.WORLD, Frame.POMDP_SPACE,
                                  region_origin=origin,
                                  search_space_resolution=self._search_space_res)
            self._octree_belief[ObjectState(OBJID, "OBJ", pomdp_point, res=1)] = VAL
        print("points incorporated to octree belief")

    def run(self):
        rospy.spin()


def main():
    rospy.init_node("spot_octree_belief_publisher")
    parser = argparse.ArgumentParser("Octree belief publisher from point cloud")
    parser.add_argument("point_cloud_topic", type=str, help="input point cloud to convert")
    parser.add_argument("--size", type=int, help="size of one dimension of the space that the octree covers."
                        "Must be a power of two. Recommended: 16, 32, or 64.", required=True)
    parser.add_argument("--res", type=float, help="meter of a side of a ground-level cube in the octree.",
                        required=True)
    parser.add_argument("--viz-topic", type=str, help="topic to publish visualization of the octree belief",
                        default="octree_belief_visual")
    args, _ = parser.parse_known_args()

    bgen = SpotOctreeBeliefPublisher(args)
    bgen.run()


if __name__ == "__main__":
    main()
