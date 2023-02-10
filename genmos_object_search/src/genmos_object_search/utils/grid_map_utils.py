import random
from collections import deque

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
