#!/usr/bin/env python
#
# Generates octree belief from given point cloud
#
# Issues (as of 08/10/2022)
#
# - Prior is hard coded and it DOES NOT SEEM TO WORK.
#   (I try to set prior at tall grids to be 0 but
#    it has no effect; Also, we need a way to automatically
#    deal with rectangular search space by setting prior
#    appropriately)
# - Probability as alpha for marker is hacky
# - Larger OctNodes always have higher probability and
#    blocks the nodes of smaller OctNodes
#    However, this is not occupancy grid map, so
#    not displaying the larger OctNodes may discard
#    information (Basically, how to make the visualization
#    clearer and better)
# - Also, it looks shifted and buggy

import argparse
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from sloop_mos_ros.conversion import convert, Frame
from sloop_object_search.oopomdp import ObjectState
from sloop_object_search.oopomdp.models.octree_belief import Octree, OctreeBelief
from sloop_object_search.utils.misc import hash16
from sloop_object_search.utils.math import clip

OBJID = "obj"

VAL = 100


def load_prior_belief(prior_data,
                      region_origin, search_space_resolution):
    """
    The prior data should be formatted as:
    objid:
        regions:
            resolution_level: 1     // resolution level of this region
            pose: [x, y, z]         // the center of the region, in world frame, (at the ground level)
            belief: e.g. 10000      // unnormalized belief about this region.
    """
    prior = {}   # objid -> {(x,y,z,r) -> value}
    for objid in prior_data:
        prior[objid] = {}
        for region in prior_data[objid]["regions"]:
            world_pos = region["pose"]
            reslevel = region["resolution_level"]
            belief = region["belief"]

            # Convert world pose to POMDP pose
            pomdp_pos = list(convert(world_pos, Frame.WORLD, Frame.POMDP_SPACE,
                                     region_origin=region_origin,
                                     search_space_resolution=search_space_resolution))
            # scale by resolution level
            pomdp_pos[0] = pomdp_pos[0] // reslevel
            pomdp_pos[1] = pomdp_pos[1] // reslevel
            pomdp_pos[2] = pomdp_pos[2] // reslevel
            octree_voxel = tuple(pomdp_pos[:3] + [reslevel])
            prior[objid][octree_voxel] = belief
    return prior


class SpotOctreeBeliefPublisher:
    def __init__(self, args):
        point_cloud_topic = args.point_cloud_topic
        self._cloud_sub = rospy.Subscriber(point_cloud_topic, PointCloud2, self._cloud_cb)
        octree = Octree((args.size, args.size, args.size))
        self._octree_belief = OctreeBelief(args.size, args.size, args.size,
                                           OBJID, "OBJ", octree)
        self._search_space_res = args.res
        self._octbelief_markers_pub = rospy.Publisher(args.viz_topic, MarkerArray, queue_size=10)

    def _cloud_cb(self, pcl2msg):
        print(pcl2msg.header.frame_id)
        pc = ros_numpy.numpify(pcl2msg)
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        origin = np.min(points, axis=0)

        # # get prior
        # # TODO: Hard coded for test. Assign zero probability to positions
        # # that are too high.
        # prior = load_prior_belief({
        #     OBJID: {
        #         "regions": [
        #             {
        #                 "pose": [0.0, 0.0, 2.0],
        #                 "belief": 0,
        #                 "resolution_level": 16
        #             },
        #             {
        #                 "pose": [-1.0, 0.0, 2.0],
        #                 "belief": 0,
        #                 "resolution_level": 16
        #             },
        #             {
        #                 "pose": [1.0, 0.0, 2.0],
        #                 "belief": 0,
        #                 "resolution_level": 16
        #             },
        #             {
        #                 "pose": [1.0, -1.0, 2.0],
        #                 "belief": 0,
        #                 "resolution_level": 16
        #             },
        #             {
        #                 "pose": [1.0, 1.0, 2.0],
        #                 "belief": 0,
        #                 "resolution_level": 16
        #             }
        #         ]
        #     },
        # }, origin, self._search_space_res)
        # for octree_voxel in prior[OBJID]:
        #     state = ObjectState(OBJID, "OBJ", octree_voxel[:3], res=octree_voxel[3])
        #     prob = prior[OBJID][octree_voxel]
        #     self._octree_belief.assign(state, prob)

        # Process points
        for p in points:
            pomdp_point = convert(p, Frame.WORLD, Frame.POMDP_SPACE,
                                  region_origin=origin,
                                  search_space_resolution=self._search_space_res)


            self._octree_belief[ObjectState(OBJID, "OBJ", pomdp_point, res=1)] = VAL
        print("points incorporated to octree belief")

        voxels = self._octree_belief.octree.collect_plotting_voxels()
        vp = [v[:3] for v in voxels]
        vr = [v[3] for v in voxels]
        probs = [self._octree_belief._probability(*Octree.increase_res(v[:3], 1, v[3]),
                                                  v[3])
                 for v in voxels]

        markers = []
        header = Header(stamp=rospy.Time.now(),
                        frame_id=pcl2msg.header.frame_id)
        for i in range(len(vp)):
            # if vr[i] > 1:
            #     # only plot if this is a leaf. So skip if otherwise.
            #     continue

            pos = convert(vp[i], Frame.POMDP_SPACE, Frame.WORLD,
                          region_origin=origin,
                          search_space_resolution=self._search_space_res)
            res = vr[i] * self._search_space_res  # resolution in meters
            prob = probs[i]
            marker = self.make_octnode_marker_msg(pos, res, prob, header)
            markers.append(marker)

        markers_array = MarkerArray(markers=markers)
        self._octbelief_markers_pub.publish(markers_array)
        print("Published octree belief markers!")


    def make_octnode_marker_msg(self, pos, res, prob, header):
        """
        Creates an rviz marker for a OctNode, specified
        by the given 3D position (in frame of header),
        resolution (in meters), and with transparency determined by
        given probability.
        """
        marker = Marker()
        marker.header = header
        marker.id = hash16((*pos, res))
        marker.type = Marker.CUBE
        marker.pose.position = Point(x=pos[0] + res/2,
                                     y=pos[1] + res/2,
                                     z=pos[2] + res/2)
        marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker.scale = Vector3(x=res, y=res, z=res)
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(1.0)
        marker.color = ColorRGBA(r=0.0, g=0.8, b=0.0, a=prob*500)
        return marker

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
