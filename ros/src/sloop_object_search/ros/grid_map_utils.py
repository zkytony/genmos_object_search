import rospy
import random
from collections import deque
from sloop_object_search.msg import GridMap2d, GridMapLoc
from sloop_object_search.oopomdp.models.grid_map import GridMap

def grid_map_to_ros_msg(grid_map, stamp=None):
    """
    Given grid_map (GridMap), return a GridMap2d message
    """
    if stamp is None:
        stamp = rospy.Time.now()
    grid_map_msg = GridMap2d()
    grid_map_msg.stamp = stamp
    grid_map_msg.width = grid_map.width
    grid_map_msg.length = grid_map.length
    grid_map_msg.name = grid_map.name
    grid_map_msg.metric_gx_min = grid_map.ranges_in_metric[0][0]
    grid_map_msg.metric_gx_max = grid_map.ranges_in_metric[0][1]
    grid_map_msg.metric_gy_min = grid_map.ranges_in_metric[1][0]
    grid_map_msg.metric_gy_max = grid_map.ranges_in_metric[1][1]
    grid_map_msg.grid_size = grid_map.grid_size
    # We just keep obstacles and free spaces
    locations = []
    for pos in grid_map.obstacles:
        locmsg = GridMapLoc()
        locmsg.loc.x = pos[0]
        locmsg.loc.y = pos[1]
        locmsg.type = "obstacle"
        locmsg.label = grid_map.labels.get(pos, "")
        locations.append(locmsg)
    for pos in grid_map.free_locations:
        locmsg = GridMapLoc()
        locmsg.loc.x = pos[0]
        locmsg.loc.y = pos[1]
        locmsg.type = "free"
        locmsg.label = grid_map.labels.get(pos, "")
        locations.append(locmsg)
    grid_map_msg.locations = locations
    return grid_map_msg


def ros_msg_to_grid_map(grid_map_msg):
    obstacles = set((locmsg.loc.x, locmsg.loc.y) for locmsg in grid_map_msg.locations
                    if locmsg.type == "obstacle")
    free_locations = set((locmsg.loc.x, locmsg.loc.y) for locmsg in grid_map_msg.locations
                         if locmsg.type == "free")
    labels = {(locmsg.loc.x, locmsg.loc.y): set(locmsg.label)
              for locmsg in grid_map_msg.locations}
    ranges_in_metric = ((grid_map_msg.metric_gx_min, grid_map_msg.metric_gx_max),
                        (grid_map_msg.metric_gy_min, grid_map_msg.metric_gy_max))
    return GridMap(grid_map_msg.width,
                   grid_map_msg.length,
                   obstacles,
                   free_locations=free_locations,
                   name=grid_map_msg.name,
                   grid_size=grid_map_msg.grid_size,
                   ranges_in_metric=ranges_in_metric,
                   labels=labels)

def cells_with_minimum_distance_from_obstacles(grid_map, dist=1):
    """
    Returns a set of locations on the given grid_map that
    are free locations and each satisfies a minimum distance
    from any obstacle in the grid map.
    """
    def _neighbors(p, d=1):
        # note that p is 3D, but we only care about x and y
        return set((p[0] + dx, p[1] + dy)
                   for dx in range(-d, d+1)
                   for dy in range(-d, d+1)
                   if not (dx == 0 and dy == 0))

    # This is implemented like a flood-fill algorithm.
    free_locs_not_considered = set(grid_map.free_locations)
    desired_cells = set()
    while len(free_locs_not_considered) > 0:
        seed_loc = random.sample(free_locs_not_considered, 1)[0]
        # BFS
        worklist = deque([seed_loc])
        visited = set({seed_loc})
        while len(worklist) > 0:
            loc = worklist.popleft()
            # Check whether
            no_obstacles_zone = _neighbors(loc, d=max(1, dist))
            if not any(p in grid_map.obstacles for p in no_obstacles_zone):
                desired_cells.add(loc)
            for neighbor_loc in no_obstacles_zone:
                if neighbor_loc not in visited:
                    if neighbor_loc in grid_map.free_locations:
                        worklist.append(neighbor_loc)
                        visited.add(neighbor_loc)
        free_locs_not_considered -= visited
        assert grid_map.free_locations != free_locs_not_considered
    return desired_cells


def obstacles_around_free_locations(grid_map, dist=1):
    """
    Returns a set of locations on the given grid map that are
    obstacles, and each is within a minimum distance from
    any free location in the grid map.
    """
    def _neighbors(p, d=1):
        # note that p is 3D, but we only care about x and y
        return set((p[0] + dx, p[1] + dy)
                   for dx in range(-d, d+1)
                   for dy in range(-d, d+1)
                   if not (dx == 0 and dy == 0))

    # This is implemented like a flood-fill algorithm.
    obstacle_locs_not_considered = set(grid_map.obstacles)
    desired_cells = set()
    while len(obstacle_locs_not_considered) > 0:
        seed_loc = random.sample(obstacle_locs_not_considered, 1)[0]
        # BFS
        worklist = deque([seed_loc])
        visited = set({seed_loc})
        while len(worklist) > 0:
            loc = worklist.popleft()
            # Check whether
            has_free_zone = _neighbors(loc, d=max(1, dist))
            if any(p in grid_map.free_locations for p in has_free_zone):
                desired_cells.add(loc)
            for neighbor_loc in has_free_zone:
                if neighbor_loc not in visited:
                    if neighbor_loc in grid_map.obstacles:
                        worklist.append(neighbor_loc)
                        visited.add(neighbor_loc)
        obstacle_locs_not_considered -= visited
        assert grid_map.obstacles != obstacle_locs_not_considered
    return desired_cells
