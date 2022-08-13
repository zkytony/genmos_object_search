

def search_region_from_occupancy_grid():
    pass


def search_region_from_point_cloud(point_cloud, world_origin=None, is_3d=False, **kwargs):
    """
    The points in the given point cloud should correspond to static
    obstacles in the environment. The extent of this point cloud forms
    the extent of the search region.

    If the point_cloud is to be projected down to 2D, then we assume
    there is a "floor" (or flat plane) in the environment. "floor_cut"
    in kwargs specifies the height below which the points are considered
    to be part of the floor.

    'world_origin' is a point in the world frame that corresponds to (0,0) or
    (0,0,0) in the POMDP model of the world. If it is None, then the world_origin
    will be set to the point with minimum coordinates in the point cloud.

    Args:
        point_cloud (common_pb2.PointCloud): The input point cloud
        is_3d (bool): whether the search region will be 3D
    """
    if is_3d:
        pass
    else:
