# Copyright 2022 Kaiyu Zheng
#
# Usage of this file is licensed under the MIT License.

import heapq
import numpy as np
from collections import deque
from genmos_object_search.utils.math import in_square

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.

      This implementation is from the pacman_assignment
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def __iter__(self):
        return iter(self.items)

    @property
    def items(self):
        return [entry[2] for entry in self.heap]


def flood_fill_2d(grid_points, seed_point, grid_brush_size=2, flood_region_size=None):
    """
    Given a numpy array of points that are supposed to be on a grid map,
    and a "seed point", flood fill by adding more grid points that are
    in empty locations to the set of grid points, starting from the seed
    point. Will not modify 'grid_points' but returns a new array.

    Args:
        grid_points (np.ndarray): a (n,2) or (n,3) array
        seed_point (np.ndarray or tuple): dimension should match that of a grid point
        grid_brush_size (int): The length (number of grids) of a square brush
            which will be used to fill out the empty spaces.
        flood_region_size (float): the maximum size (number of grids) of
            the flooding region which is a square.
    """
    def _neighbors(p, d=1):
        # this works with both 2D or 3D points
        return set((p[0] + dx, p[1] + dy, *p[2:])
                   for dx in range(-d, d+1)
                   for dy in range(-d, d+1)
                   if not (dx == 0 and dy == 0))

    seed_point = tuple(seed_point)
    if grid_points.shape[1] != len(seed_point):
        raise ValueError("grid points and seed point have different dimensions.")

    grid_points_set = set(map(tuple, grid_points))
    # BFS
    worklist = deque([seed_point])
    visited = set({seed_point})
    new_points = set()
    while len(worklist) > 0:
        point = worklist.popleft()
        # Imagine placing a square brush centered at the point.
        # We assume that the point always represents a valid free cell
        brush_points = _neighbors(point, d=max(1, grid_brush_size//2))
        new_points.update(brush_points)
        if not any(bp in grid_points_set for bp in brush_points):
            # This brush stroke fits; we will consider all brush points
            # as potential free cells - i.e. neighbors
            for neighbor_point in brush_points:
                if neighbor_point not in visited:
                    if flood_region_size is not None:
                        if not in_square(neighbor_point, seed_point, flood_region_size):
                            continue  # skip this point: too far.
                    worklist.append(neighbor_point)
                    visited.add(neighbor_point)
    return np.array(list(grid_points_set | new_points))
