#!/usr/bin/env python
# See implementation of the grid map publisher at sloop_mos_spot/grid_map_publisher.py
import rospy
import argparse
from sloop_mos_spot.grid_map_publisher import GraphNavPointCloudToGridMapPublisher

def main():
    parser = argparse.ArgumentParser("PointCloud2 to GridMap2d")
    parser.add_argument("--point-cloud-topic", type=str, help="name of point cloud topic to subscribe to",
                        default="/graphnav_map_publisher/graphnav_points")
    parser.add_argument("--waypoint-topic", type=str, help="name of the topic for GraphNav waypoints",
                        default="/graphnav_waypoints")
    parser.add_argument("--grid-map-topic", type=str, help="name of grid map topic to publish at",
                        default="/graphnav_gridmap")
    parser.add_argument("--grid-size", type=float, help="grid size (m). Default: 0.25",
                        default=0.25)
    parser.add_argument("--layout-cut", type=float, help="the z level below which obstacles are ignored",
                        default=0.65)
    parser.add_argument("--updating", action="store_true",
                        help="Keeps subscribing to point cloud and update the grid map; Otherwise, publishes once and latches.")
    parser.add_argument("--name", type=str, help="name of the grid map",
                        required=True)
    parser.add_argument("--debug", action="store_true", help="Debug grid map generation")
    args, _ = parser.parse_known_args()
    gmpub = GraphNavPointCloudToGridMapPublisher(args)
    rate = rospy.Rate(5)
    while not gmpub.published:
        rate.sleep()
    rospy.loginfo("grid map publisher done!")

if __name__ == "__main__":
    main()
