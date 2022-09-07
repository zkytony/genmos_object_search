import open3d as o3d
import numpy as np
import random
from sloop_object_search.utils.colors import color_map, cmaps, lighter
from sloop_object_search.utils import math as math_utils
from sloop_object_search.oopomdp.models.octree_belief import RegionalOctreeDistribution, Octree
from sloop_object_search.oopomdp.models.search_region import SearchRegion3D, SearchRegion2D

def cube_unfilled(scale=1):
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
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set

# def _align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
#     """
#     Aligns vector a to vector b with axis angle rotation

#     https://github.com/isl-org/Open3D/pull/738#issuecomment-564785941
#     """
#     if np.array_equal(a, b):
#         return None, None
#     axis_ = np.cross(a, b)
#     axis_ = axis_ / np.linalg.norm(axis_)
#     angle = np.arccos(np.dot(a, b))

#     return axis_, angle

# def line_between_thick(pos1, pos2, radius=0.15):
#     length = euclidean_dist(pos1, pos2)
#     cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
#     cylinder.translate()


def line_between(pos1, pos2):
    points = [pos1, pos2]
    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set


def cube_filled(scale=1, color=[1,0,0], alpha=1.0, name=None, with_border=False):
    """Note : This doesn't return a regular TriangleMesh. To
    draw this, need to call open3d.visualization.draw. May not
    be compatible with other draws"""
    box = o3d.geometry.TriangleMesh.create_box()
    box.scale(scale, center=box.get_center())
    box.paint_uniform_color(color)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = np.array([1, 1, 1, alpha])
    if name is None:
        name = "cube{}".format(random.randint(1000, 200000))#"cubefilled-{}".format(color + [alpha])
    return {'name': name, 'geometry': box, 'material': mat}


def draw_search_region3d(search_region, octree_dist=None, points=None,
                         color_by_prob=True, cmap=cmaps.COLOR_MAP_HALLOWEEN):
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
    probs = [octree_dist.prob_at(
        *Octree.increase_res(v[:3], 1, v[3]), v[3])
             for v in voxels]
    for i in range(len(vp)):
        pos = search_region.to_world_pos(vp[i])
        size = vr[i] * search_region.search_space_resolution  # cube size in meters
        mesh_box = cube_unfilled(scale=size)
        mesh_box.translate(np.asarray(pos))
        if color_by_prob:
            color = color_map(probs[i], [min(probs), max(probs)], cmap)
        else:
            color = [0.9, 0.1, 0.1]
        mesh_box.paint_uniform_color(color)
        geometries.append(mesh_box)
    o3d.visualization.draw_geometries(geometries)
    return geometries

def draw_octree_dist(octree_dist, viz=True, color_by_prob=True,
                     cmap=cmaps.COLOR_MAP_GRAYS):
    """draw the octree dist in POMDP space."""
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4.0, origin=[0.0, 0.0, 0.0])
    geometries = [mesh_frame]
    # visualize octree
    voxels = octree_dist.collect_plotting_voxels()
    vp = [v[:3] for v in voxels]
    vr = [v[3] for v in voxels]  # resolutions
    probs = [octree_dist.prob_at(
        *Octree.increase_res(v[:3], 1, v[3]), v[3])
             for v in voxels]
    for i in range(len(vp)):
        if color_by_prob:
            color = color_map(probs[i], [min(probs), max(probs)], cmap)
            alpha = math_utils.remap(probs[i], min(probs)-0.00001, max(probs), 0.001, 0.8)
        else:
            color = [0.9, 0.1, 0.1]
            alpha = 0.6
        pos = vp[i]
        size = vr[i]  # cube size in meters
        mesh_box = cube_unfilled(scale=size)
        mesh_box.translate(np.asarray(pos))
        mesh_box.paint_uniform_color(color)
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

def draw_locdist2d(locdist2d,
                   marker_height=0.1, color=[0.2, 0.7, 0.2],
                   search_region=None, viz=True):
    """locdist2d (LocDist2D). Uses the algorithm I always use
    to draw 2D belief. If search_region is provided, will draw
    the positions in world frame"""
    prob_max = max(locdist2d.values_dict.values())
    prob_min = min(locdist2d.values_dict.values())
    hist = {}
    for loc in locdist2d.values_dict:
        hist[loc] = locdist2d.prob_at(*loc)

    if search_region is None:
        grid_size = 1
    else:
        grid_size = search_region.grid_size

    geometries = []
    last_val = -1
    i = 0
    for pos in reversed(sorted(hist, key=hist.get)):
        if last_val != -1:
            color = lighter(np.asarray(color)*255, 1-hist[pos]/last_val)/255

        stop = np.mean(np.array(color[:3]) / np.array([255, 255, 255])) > 0.999
        if not stop:
            marker = o3d.geometry.TriangleMesh.create_cylinder(grid_size/2, marker_height)
            if search_region is None:
                marker.translate(np.asarray([*pos, 0.0]))
            else:
                pos_world = search_region.to_world_pos(pos)
                marker.translate(np.asarray([*pos_world, 0.0]))
            marker.paint_uniform_color(np.asarray(color))
            geometries.append(marker)
            last_val = hist[pos]
            i += 1
            if last_val <= 0:
                break
    if viz:
        o3d.visualization.draw_geometries(geometries)
    return geometries


def draw_robot_pose(robot_pose):
    # The robot by default looks at +x direction with zero quaternion
    # but the arrow points to +z with zero quaternion. So, we will
    # have to do the following to align the two
    default_o3d_rotation = [0,90,0]  # don't change this

    rotation = math_utils.quat_to_euler(*robot_pose[3:])
    position = robot_pose[:3]
    sensor_pose = (*position, *math_utils.euler_to_quat(*rotation))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.5, cone_radius=0.75, cylinder_height=3.0,
        cone_height=1.8)

    _o3d_rotation = np.array(default_o3d_rotation) + np.array(rotation)
    arrow.rotate(math_utils.R_euler(*_o3d_rotation).as_matrix())
    arrow.translate(np.asarray(sensor_pose[:3]))
    arrow.paint_uniform_color([0.4, 0.4, 0.4])
    return arrow

def draw_fov(visible_volume, obstacles_hit=None):
    geometries = []
    for voxel in visible_volume:
        if voxel not in obstacles_hit:
            x, y, z, r = voxel
            box = o3d.geometry.TriangleMesh.create_box(width=r, height=r, depth=r)
            box.translate(np.asarray([x*r,y*r,z*r]))
            box.paint_uniform_color([0.0, 0.55, 0.98])
            geometries.append(box)

    if obstacles_hit is not None:
        for voxel in obstacles_hit:
            x, y, z, r = voxel
            box = o3d.geometry.TriangleMesh.create_box(width=r, height=r, depth=r)
            box.translate(np.asarray([x*r,y*r,z*r]))
            box.paint_uniform_color([0.0, 0.05, 0.75])
            geometries.append(box)
    return geometries


def draw_topo_graph3d(topo_map,
                      search_region,
                      object_beliefs=None,
                      node_color=[0.99, 0.6, 0.2],
                      edge_color=[0.05, 0.04, 0.7],
                      viz=True):
    occupancy_octree = search_region.octree_dist
    geometries = []
    # geometries = draw_octree_dist(occupancy_octree, viz=False)

    if object_beliefs is not None:
        for objid in object_beliefs:
            geometries.extend(draw_octree_dist(object_beliefs[objid].octree_dist, viz=False,
                                               cmap=cmaps.COLOR_MAP_GRAYS))

    for nid in topo_map.nodes:
        node = topo_map.nodes[nid]
        x, y, z = node.pos
        r = 1
        box = o3d.geometry.TriangleMesh.create_box(width=r, height=r, depth=r)
        box.translate(np.asarray([x*r,y*r,z*r]))
        box.paint_uniform_color(node_color)
        geometries.append(box)

    for eid in topo_map.edges:
        edge = topo_map.edges[eid]
        if not edge.degenerate:
            node1, node2 = edge.nodes
            pos1 = math_utils.originbox_to_centerbox((node1.pos, 1, 1, 1))[0]
            pos2 = math_utils.originbox_to_centerbox((node2.pos, 1, 1, 1))[0]
            line = line_between(pos1, pos2)
            line.paint_uniform_color(edge_color)
            geometries.append(line)
    if viz:
        o3d.visualization.draw_geometries(geometries, width=760, height=480)
    return geometries

def draw_fov_2d(cells, robot_pose, viz=False):
    """cells is a set of ((x,y), label) tuples"""
    robot_pose3d = (*robot_pose[:2], 0.0, *math_utils.euler_to_quat(0, 0, robot_pose[2]))
    pose_arrow = draw_robot_pose(robot_pose3d)

    pcd2 = o3d.geometry.PointCloud()
    cell_points = np.array([[*cell[0], 0.0] for cell in cells])
    pcd2.points = o3d.utility.Vector3dVector(cell_points)
    pcd2.colors = o3d.utility.Vector3dVector(np.full((len(cell_points), 3), (0.2, 0.2, 0.6)))

    geometries = [pcd2, pose_arrow]
    if viz:
        o3d.visualization.draw_geometries(geometries, width=760, height=480)
    return geometries
