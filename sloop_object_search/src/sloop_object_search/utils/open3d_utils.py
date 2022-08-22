import open3d as o3d
import numpy as np
from sloop_object_search.oopomdp.models.octree_belief import RegionalOctreeDistribution
from sloop_object_search.oopomdp.models.search_region import SearchRegion3D, SearchRegion2D

def cube_unfilled(scale=1, color=[1,0,0]):
    # http://www.open3d.org/docs/0.9.0/tutorial/Basic/visualization.html
    if hasattr(scale, "__len__"):
        scale_x, scale_y, scale_z = scale
    else:
        scale_x = scale_y = scale_z = scale

    points = [
        [0,        0,       0],
        [scale_x,  0,       0],
        [0,        scale_y, 0],
        [scale_x,  scale_y, 0],
        [0,        0,       scale_z],
        [scale_x,  0,       scale_z],
        [0,        scale_y, scale_z],
        [scale_x,  scale_y, scale_z],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set


def draw_search_region3d(search_region, octree_dist=None, points=None):
    """
    By default draws the octree_dist in search_region, unless one
    is provided.
    points: a numpy array (N, 3) that is the point cloud
    associated with creating this search region.
    """
    if not isinstance(search_region, SearchRegion3D):
        raise TypeError("search region must be SearchRegion3D")
    if octree_dist is None:
        octree_dist = search_region.octree_dist

    # Draw region box
    region = octree_dist.region
    origin, w, l, h = region
    origin = search_region.to_world_pos(origin)
    sizes = np.asarray([w, l, h]) * search_region.search_space_resolution
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.2, origin=origin)
    region_box = cube_unfilled(scale=sizes)
    region_box.translate(np.asarray(origin))
    region_box.paint_uniform_color([0.1, 0.9, 0.1])
    geometries = [mesh_frame, region_box]

    # Will visualize both the point cloud and the octree
    # visualize point cloud
    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), (0.8, 0.8, 0.8)))
        geometries.append(pcd)

    # visualize octree
    voxels = octree_dist.octree.collect_plotting_voxels()
    vp = [v[:3] for v in voxels]
    vr = [v[3] for v in voxels]  # resolutions
    vv = [v[4] for v in voxels]  # values
    for i in range(len(vp)):
        pos = search_region.to_world_pos(vp[i])
        size = vr[i] * search_region.search_space_resolution  # cube size in meters
        mesh_box = cube_unfilled(scale=size)
        mesh_box.translate(np.asarray(pos))
        mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        geometries.append(mesh_box)
    o3d.visualization.draw_geometries(geometries)
    return geometries

def draw_octree_dist(octree_dist, viz=True):
    """draw the octree dist in POMDP space."""
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4.0, origin=[0.0, 0.0, 0.0])
    geometries = [mesh_frame]
    # visualize octree
    voxels = octree_dist.octree.collect_plotting_voxels()
    vp = [v[:3] for v in voxels]
    vr = [v[3] for v in voxels]  # resolutions
    vv = [v[4] for v in voxels]  # values
    for i in range(len(vp)):
        pos = vp[i]
        size = vr[i]  # cube size in meters
        mesh_box = cube_unfilled(scale=size)
        mesh_box.translate(np.asarray(pos))
        mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        geometries.append(mesh_box)
    if viz:
        o3d.visualization.draw_geometries(geometries)
    return geometries


def draw_octree_dist_in_search_region(octree_dist, search_region, points=None):
    return draw_search_region3d(search_region, octree_dist=octree_didst, points=points)


def draw_search_region2d(search_region, grid_robot_position=None, points=None):

    if not isinstance(search_region, SearchRegion2D):
        raise TypeError("search region must be SearchRegion2D")

    geometries = []

    if points is not None:
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(np.asarray(list(points)))
        pcd0.colors = o3d.utility.Vector3dVector(np.full((len(points), 3), (0.8, 0.8, 0.8)))
        geometries.append(pcd0)

    grid_map = search_region.grid_map

    pcd1 = o3d.geometry.PointCloud()
    freeloc_points = np.asarray(list(search_region.to_world_pos(loc)
                                     for loc in grid_map.free_locations))
    freeloc_points = np.append(freeloc_points, np.zeros((len(freeloc_points), 1)), axis=1)
    pcd1.points = o3d.utility.Vector3dVector(freeloc_points)
    pcd1.colors = o3d.utility.Vector3dVector(np.full((len(grid_map.free_locations), 3), (0.8, 0.8, 0.8)))
    geometries.append(pcd1)

    pcd2 = o3d.geometry.PointCloud()
    obloc_points = np.asarray(list(search_region.to_world_pos(loc)
                                   for loc in grid_map.obstacles))
    obloc_points = np.append(obloc_points, np.zeros((len(obloc_points), 1)), axis=1)
    pcd2.points = o3d.utility.Vector3dVector(obloc_points)
    pcd2.colors = o3d.utility.Vector3dVector(np.full((len(grid_map.obstacles), 3), (0.2, 0.2, 0.2)))
    if grid_robot_position is not None:
        pcd2.points.append([*search_region.to_world_pos(grid_robot_position), 1])
        pcd2.colors.append([0.0, 0.8, 0.0])
    geometries.append(pcd2)
    o3d.visualization.draw_geometries(geometries)
    return geometries
