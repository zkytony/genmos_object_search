#!/usr/bin/env python
#
# Generates octree belief from given point cloud
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

OBJID = "obj"

VAL = 100

def change_res(point, r1, r2):
    x,y,z = point
    return (x // (r2 // r1), y // (r2 // r1), z // (r2 // r1))


class SpotOctreeBeliefPublisher:
    def __init__(self, args):
        point_cloud_topic = args.point_cloud_topic
        self._cloud_sub = rospy.Subscriber(point_cloud_topic, PointCloud2, self._cloud_cb)
        octree = Octree(OBJID, (args.size, args.size, args.size))
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
        for p in points:
            pomdp_point = convert(p, Frame.WORLD, Frame.POMDP_SPACE,
                                  region_origin=origin,
                                  search_space_resolution=self._search_space_res)
            self._octree_belief[ObjectState(OBJID, "OBJ", pomdp_point, res=1)] = VAL
        print("points incorporated to octree belief")

        voxels = self._octree_belief.octree.collect_plotting_voxels()
        vp = [v[:3] for v in voxels]  # position in the world frame
        vr = [v[3] for v in voxels]
        probs = [self._octree_belief._probability(*change_res(v[:3], 1, v[3]),
                                                  v[3])
                 for v in voxels]

        markers = []
        header = Header(stamp=rospy.Time.now(),
                        frame_id=pcl2msg.header.frame_id)
        for i in range(len(vp)):
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
        marker.pose.position = Point(*pos)
        marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker.scale = Vector3(x=res, y=res, z=res)
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.25)
        marker.color = ColorRGBA(r=0.0, g=0.8, b=0.0, a=prob*100)
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
