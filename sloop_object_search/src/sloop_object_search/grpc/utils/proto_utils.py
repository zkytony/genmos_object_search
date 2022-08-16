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
    if search_region_params_2d_pb.HasField('layout_cut'):
        params["layout_cut"] = search_region_params_2d_pb.layout_cut
    if search_region_params_2d_pb.HasField('floor_cut'):
        params["floor_cut"] = search_region_params_2d_pb.floor_cut
    if search_region_params_2d_pb.HasField('grid_size'):
        params["grid_size"] = search_region_params_2d_pb.grid_size
    if search_region_params_2d_pb.HasField('brush_size'):
        params["brush_size"] = search_region_params_2d_pb.brush_size
    if search_region_params_2d_pb.HasField('region_size'):
        params["region_size"] = search_region_params_2d_pb.region_size
    if search_region_params_2d_pb.HasField('debug'):
        params["debug"] = search_region_params_2d_pb.debug
    return params

def process_search_region_params_3d(search_region_params_3d_pb):
    params = {}
    if search_region_params_3d_pb.HasField('octree_size'):
        params["octree_size"] = search_region_params_3d_pb.octree_size
    if search_region_params_3d_pb.HasField('search_space_resolution'):
        params["search_space_resolution"] = search_region_params_3d_pb.search_space_resolution
    if search_region_params_3d_pb.HasField('debug'):
        params["debug"] = search_region_params_3d_pb.debug
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
        point_pb = PointCloud.Point(pos=Vec3(x=p[0], y=p[1], z=p[2]))
        points_pb.append(point_pb)

    header = Header(stamp=Timestamp().GetCurrentTime(),
                    frame_id=cloud_msg.header.frame_id)
    cloud_pb = PointCloud(header=header,
                          points=points_pb)
    return cloud_pb


def pointcloudproto_to_array(point_cloud):
    points_array = np.array([[p.pos.x, p.pos.y, p.pos.z]
                             for p in point_cloud.points])
    return points_array


def make_header(frame_id=None, stamp=None):
    if stamp is None:
        stamp = Timestamp().GetCurrentTime()
    if frame_id is None:
        return Header(stamp=stamp)
    else:
        return Header(stamp=stamp, frame_id=frame_id)

def interpret_robot_pose(request):
    """Given a request proto object whose definition contains

        oneof robot_pose {
            Pose2D robot_pose_2d = 4;
            Pose3D robot_pose_3d = 5;
          }

    return a tuple representation of the pose. If it is 2D,
    then return (x, y, th). If it is 3D, then return (x, y, z,
    qx, qy, qz, qw).
    """
    if request.HasField('robot_pose_2d'):
        robot_pose = (request.robot_pose_2d.x,
                      request.robot_pose_2d.y,
                      request.robot_pose_2d.th)
    elif request.HasField('robot_pose_3d'):
        robot_pose = (request.robot_pose_3d.position.x,
                      request.robot_pose_3d.position.y,
                      request.robot_pose_3d.position.z,
                      request.robot_pose_3d.rotation.x,
                      request.robot_pose_3d.rotation.y,
                      request.robot_pose_3d.rotation.z,
                      request.robot_pose_3d.rotation.w,)
    else:
        raise ValueError("request does not contain valid robot pose field.")
    return robot_pose
