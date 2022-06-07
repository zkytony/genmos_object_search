# Copyright 2022 Kaiyu Zheng
#
# Usage of this file is licensed under the MIT License.

import copy
import math
import json
import sys
import numpy as np
from collections import deque
from sloop_object_search.utils.math import remap, to_degrees, euclidean_dist

def neighbors(x,y):
    return [(x+1, y), (x-1,y),
            (x,y+1), (x,y-1)]

class GridMap:
    """A grid map is a collection of locations, walls, and some locations have obstacles.
    Horizontal axis is x, vertical axis is y.

    Note that the coordinates in grid map starts from (0,0) to (width-1,length-1).
    """

    def __init__(self, width, length, obstacles,
                 unknown=None, name="grid_map"):
        """
        xpos (list): list of x coordinates for free cells
        ypos (list): list of y coordinates for free cells
        obstacles (set): a set of locations for the obstacles
        unknown (set): locations that have unknown properties.
            If None, then this set will be empty; The free locations
            of a grid map is
                ALL_CELLS(width, length) - obstacles - unknown
        """
        self.width = width
        self.length = length
        self.name = name
        self.update(obstacles, unknown=unknown)

        # Caches the computations
        self._geodesic_dist_cache = {}
        self._blocked_cache = {}

        # Labels on grid cells
        self._labels = {}  # maps from position (x,y) to a set of labels

    def __contains__(self, loc):
        return loc in self.free_locations\
            or loc in self.obstacles\
            or loc in self.unknown

    def label(self, x, y, label):
        if (x,y) not in self:
            raise ValueError(f"Cell ({x}, {y}) not on grid map")
        if (x,y) not in self._labels:
            self._labels[(x,y)] = set()
        self._labels[(x,y)].add(label)

    def label_all(self, locs, label):
        for loc in locs:
            self.label(*loc, label)

    def filter_by_label(self, label):
        """Returns a set of locations in the grid map with label"""
        return {loc for loc in self.free_locations
                if label in self._labels.get(loc, set())}\
                | {loc for loc in self.obstacles
                   if label in self._labels.get(loc, set())}\
                | {loc for loc in self.unknown
                   if label in self._labels.get(loc, set())}

    def update(self, obstacles, unknown=None):
        all_positions = {(x,y) for x in range(self.width)
                         for y in range(self.length)}

        self.obstacles = obstacles
        if unknown is None:
            unknown = set()
        self.unknown = unknown
        self.free_locations = all_positions - self.obstacles - self.unknown

    def free_region(self, x, y):
        """Given (x,y) location, return a set of locations
        that are free and connected to (x,y)"""
        region = set()
        q = deque()
        q.append((x,y))
        visited = set()
        while len(q) > 0:
            loc = q.popleft()
            region.add(loc)
            for nb_loc in neighbors(*loc):
                if nb_loc in self.free_locations:
                    if nb_loc not in visited:
                        visited.add(nb_loc)
                        q.append(nb_loc)
        return region

    def boundary_cells(self, thickness=1):
        """
        Returns a set of locations corresponding to
        obstacles that lie between free space and occluded spaces.
        These are usually locations where objects are placed.
        """
        last_boundary = set()
        for i in range(thickness):
            boundary = set()
            for x, y in self.obstacles:
                for nx, ny in neighbors(x, y):
                    if (nx, ny) in self.free_locations\
                       or (nx, ny) in last_boundary:
                        boundary.add((x,y))
                        break
            last_boundary.update(boundary)
        return last_boundary

    def closest_free_cell(self, loc):
        """Snaps given loc (x,y) to the closest grid cell"""
        return min(self.free_locations,
                   key=lambda l: euclidean_dist(l, loc))

    def shortest_path(self, gloc1, gloc2):
        """
        Computes the shortest distance between two locations.
        The two locations will be snapped to the closest free cell.
        """
        def get_path(s, t, prev):
            v = t
            path = [t]
            while v != s:
                v = prev[v]
                path.append(v)
            return path

        # BFS; because no edge weight
        visited = set()
        q = deque()
        q.append(gloc1)
        prev = {gloc1:None}
        while len(q) > 0:
            loc = q.popleft()
            if loc == gloc2:
                return get_path(gloc1, gloc2, prev)
            for nb_loc in neighbors(*loc):
                if nb_loc in self.free_locations:
                    if nb_loc not in visited:
                        q.append(nb_loc)
                        visited.add(nb_loc)
                        prev[nb_loc] = loc
        return None

    def geodesic_distance(self, loc1, loc2):
        """Reference: https://arxiv.org/pdf/1807.06757.pdf
        The geodesic distance is the shortest path distance
        in the environment.

        Geodesic distance: the distance between two vertices
        in a graph is the number of edges in a shortest path.

        NOTE: This is NOT the real geodesic distance in
        the THOR environment, but an approximation for
        POMDP agent's behavior. The Unit here is No.GridCells

        This is computed by first snapping loc1, loc2
        to the closest free grid cell then find the
        shortest path on the grid between them.

        Args:
           loc1, loc2 (tuple) grid map coordinates
        """
        _key = tuple(loc1), tuple(loc2)
        if _key in self._geodesic_dist_cache:
            return self._geodesic_dist_cache[_key]
        else:
            path = self.shortest_path(loc1, loc2)
            if path is not None:
                dist = len(path)
            else:
                dist = float("inf")
            self._geodesic_dist_cache[_key] = dist
            return dist

    def blocked(self, loc1, loc2, nsteps=40):
        """
        Returns True if:
        - loc1 is not free,
        OR
        - loc1 is a reachable location AND
        - the line segment between loc1 and loc2 goes through
          an obstacle, and then goes through a free cell
          (i.e. blocked by an obstacle)

        This is checked by simulating a straightline from loc1 to loc2 and check
        if any step on the line is at an obstacle.
        """
        if loc1 == loc2:
            return False

        if loc1 not in self.free_locations:
            return True

        _key = tuple(loc1), tuple(loc2)
        if _key in self._blocked_cache:
            return self._blocked_cache[_key]

        # vec = np.array([px - rx, py - ry]).astype(float)
        # vec /= np.linalg.norm(vec)
        loc1 = np.asarray(loc1)
        x1, y1 = loc1
        x2, y2 = loc2
        vec = np.array([x2 - x1, y2 - y1]).astype(float)
        vec /= np.linalg.norm(vec)

        # Check points along the line from robot pose to the point
        status = "start"
        dist = euclidean_dist(loc1, loc2)
        step_size = dist / nsteps
        t = 0
        while t < nsteps:
            line_point = tuple(np.round(loc1 + (t*step_size*vec)).astype(int))
            if line_point in self.obstacles:
                status = "hits_obstacle"
            elif line_point in self.free_locations:
                if status == "hits_obstacle":
                    status = "blocked"
                    break
            t += 1
        result = status == "blocked"
        self._blocked_cache[_key] = result
        return result

    def save(self, savepath):
        """Saves this grid map as a json file to the save path.
        Args:
            savepath (Ste): Path to the output .json file"""

        obstacles_arr = [list(map(int, pos)) for pos in self.obstacles]
        unknown_arr = [list(map(int, pos)) for pos in self.unknown]

        output = {
            'width': int(self.width),
            'length': int(self.length),
            'obstacles': obstacles_arr,
            'unknown': unknown_arr,
            'name': self.name
        }

        if self.ranges_in_thor is not None:
            thor_gx_min, thor_gx_max = self.ranges_in_thor[0]
            thor_gy_min, thor_gy_max = self.ranges_in_thor[1]
            output['ranges_in_thor'] = [[int(thor_gx_min), int(thor_gx_max)],
                                        [int(thor_gy_min), int(thor_gy_max)]]
        else:
            output['ranges_in_thor'] = 'null'

        output['grid_size'] = self.grid_size\
            if self.grid_size is not None else "null"

        with open(savepath, 'w') as f:
            json.dump(output, f)

    @staticmethod
    def load(loadpath):
        with open(loadpath) as f:
            data = json.load(f)

        obstacles = set(map(tuple, data["obstacles"]))
        unknown = set(map(tuple, data["unknown"]))
        return GridMap(data["width"],
                       data["length"],
                       obstacles,
                       unknown=unknown,
                       name=data["name"],
                       ranges_in_thor=data["ranges_in_thor"],
                       grid_size=data["grid_size"])
