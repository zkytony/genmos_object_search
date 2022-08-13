import numpy as np

# we don't want to crash if a ros-related package is not installed.
import importlib
if importlib.util.find_spec("ros_numpy") is not None:
    import ros_numpy

from google.protobuf.timestamp_pb2 import Timestamp

from sloop_object_search.grpc.observation_pb2 import PointCloud
from sloop_object_search.grpc.common_pb2 import Vec3, Header

def process_search_region_params_2d(search_region_params_2d_pb):
    params = {}
    if search_region_params_2d_pb.layout_cut:
        params["layout_cut"] = search_region_params_2d_pb.layout_cut
    return params

def pointcloud2_to_pointcloudproto(cloud_msg):
    """
    Converts a PointCloud2 message to a PointCloud proto message.

    Args:
       cloud_msg (sensor_msgs.PointCloud2)
    """
    pcl_raw_array = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
    points_xyz_array = ros_numpy.point_cloud2.get_xyz_points(pcl_raw_array)

    points_pb = []
    for p in points_xyz_array:
        point_pb = PointCloud.Point(pos=Vec3(x=p[0], y=p[1], z=p[2]), label="")
        points_pb.append(point_pb)

    header = Header(stamp=Timestamp().GetCurrentTime(),
                    frame_id=cloud_msg.header.frame_id)
    cloud_pb = PointCloud(header=header,
                          points=points_pb)
    return cloud_pb


def pointcloudproto_to_array(point_cloud):
    return np.array([[p.pos.x, p.pos.y, p.pos.z]
                     for p in point_cloud.points])
