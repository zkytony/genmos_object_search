# Copyright 2022 Kaiyu Zheng
#
# Usage of this file is licensed under the MIT License.

import copy
import math
import json
import sys
import numpy as np
from collections import deque
from genmos_object_search.utils.math import remap, to_degrees, euclidean_dist

def neighbors(x,y):
    return [(x+1, y), (x-1,y),
            (x,y+1), (x,y-1)]

class GridMap:
    """A grid map is a collection of locations, walls, and some locations have obstacles.
    Horizontal axis is x, vertical axis is y.

    Note that the coordinates in grid map starts from (0,0) to (width-1,length-1).

    The grid map can correspond its coordinates to metric-space coordinates if parameters
    'grid_size' and 'ranges-in_metric' are provided; Adapted from thortils.GridMap.
    """

    def __init__(self, width, length, obstacles,
                 free_locations=None, unknown=None, name="grid_map",
                 ranges_in_metric=None, grid_size=None, labels=None):
        """
        obstacles (set): a set of locations for the obstacles
        unknown (set): locations that have unknown properties.
            If None, then this set will be empty; The free locations
            of a grid map is
                ALL_CELLS(width, length) - obstacles - unknown
            Ignored if free_locations is not None.
        free_locations (set): locations that are free.
            The unknown locations of a grid map is
                ALL_CELLS(width, length) - obstacles - unknown
            this has higher priority than unknown.
        labels (dict): maps from location to a set of strings
        """
        self.width = width
        self.length = length
        self.name = name
        self.ranges_in_metric = ranges_in_metric
        self.grid_size = grid_size
        self.update(obstacles, unknown=unknown, free_locations=free_locations)

        # Caches the computations
        self._geodesic_dist_cache = {}
        self._blocked_cache = {}

        # Labels on grid cells
        self.labels = labels
        if labels is None:
            self.labels = {}  # maps from position (x,y) to a set of labels

    def update(self, obstacles, unknown=None, free_locations=None):
        all_positions = {(x,y) for x in range(self.width)
                         for y in range(self.length)}

        self.obstacles = obstacles
        if free_locations is not None:
            # We need to make free_locations and obstacles disjoint sets; Overlaps
            # are considered obstacles
            overlap = self.obstacles.intersection(free_locations)
            free_locations -= overlap
            assert self.obstacles.isdisjoint(free_locations)
            self.free_locations = free_locations
            self.unknown = all_positions - self.obstacles - self.free_locations
        else:
            # We need to make unknown and obstacles disjoint sets; Overlaps
            # are considered obstacles
            if unknown is None:
                unknown = set()
            else:
                # check that obstacles and unknown locations are disjoint sets
                overlap = self.obstacles.intersection(unknown)
                unknown -= overlap
                assert self.obstacles.isdisjoint(unknown)
            self.unknown = unknown
            self.free_locations = all_positions - self.obstacles - self.unknown

    def __contains__(self, loc):
        return loc in self.free_locations\
            or loc in self.obstacles\
            or loc in self.unknown

    def to_metric_pose(self, x, y, th):
        """Given a point (x, y) in the grid map and th (degrees),
        convert it to a tuple (metric_x, metric_y, degrees_th)"""
        return (*self.to_metric_pos(x, y), self.to_metric_yaw(th))

    def to_metric_pos(self, x, y):
        """
        Given a point (x,y) in the grid map, convert it to (x,z) in
        the THOR coordinte system (grid size is accounted for).
        If grid_size is None, will return the integers
        for the corresponding coordinate.
        """
        # Note that y is z in Unity
        metric_gx_min, metric_gx_max = self.ranges_in_metric[0]
        metric_gy_min, metric_gy_max = self.ranges_in_metric[1]
        metric_gx = remap(x, 0, self.width, metric_gx_min, metric_gx_max)
        metric_gy = remap(y, 0, self.length, metric_gy_min, metric_gy_max)
        if self.grid_size is not None:
            # Snap to grid
            return (self.grid_size * round((metric_gx * self.grid_size) / self.grid_size),
                    self.grid_size * round((metric_gy * self.grid_size) / self.grid_size))
        else:
            return (metric_gx, metric_gy)

    # def to_grid_pose(self, metric_x, metric_y, metric_th, avoid_obstacle=False):
    #     return (*self.to_grid_pos(metric_x, metric_y, avoid_obstacle=avoid_obstacle),
    #             self.to_grid_yaw(metric_th))

    def to_grid_pos(self, metric_x, metric_y, avoid_obstacle=False):
        """
        Convert thor location to grid map location. If grid_size is specified,
        then will regard metric_x, metric_y as the original Unity coordinates.
        If not, then will regard them as grid indices but with origin not at (0,0).
        """
        if self.grid_size is not None:
            metric_gx = int(round(metric_x / self.grid_size))
            metric_gy = int(round(metric_y / self.grid_size))
        else:
            metric_gx = metric_x
            metric_gy = metric_y

        # remap coordinates to be nonnegative (origin AT (0,0))
        metric_gx_min, metric_gx_max = self.ranges_in_metric[0]
        metric_gy_min, metric_gy_max = self.ranges_in_metric[1]
        gx = int(remap(metric_gx, metric_gx_min, metric_gx_max, 0, self.width, enforce=True))
        gy = int(remap(metric_gy, metric_gy_min, metric_gy_max, 0, self.length, enforce=True))
        if avoid_obstacle and (gx, gy) not in self.free_locations:
            return self.closest_free_cell((gx, gy))
        else:
            return gx, gy

    def label(self, x, y, label):
        if (x,y) not in self:
            raise ValueError(f"Cell ({x}, {y}) not on grid map")
        if (x,y) not in self.labels:
            self.labels[(x,y)] = set()
        self.labels[(x,y)].add(label)

    def label_all(self, locs, label):
        for loc in locs:
            self.label(*loc, label)

    def filter_by_label(self, label):
        """Returns a set of locations in the grid map with label"""
        return {loc for loc in self.free_locations
                if label in self.labels.get(loc, set())}\
                | {loc for loc in self.obstacles
                   if label in self.labels.get(loc, set())}\
                | {loc for loc in self.unknown
                   if label in self.labels.get(loc, set())}

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

        if self.ranges_in_metric is not None:
            metric_gx_min, metric_gx_max = self.ranges_in_metric[0]
            metric_gy_min, metric_gy_max = self.ranges_in_metric[1]
            output['ranges_in_metric'] = [[int(metric_gx_min), int(metric_gx_max)],
                                        [int(metric_gy_min), int(metric_gy_max)]]
        else:
            output['ranges_in_metric'] = 'null'

        output['grid_size'] = self.grid_size\
            if self.grid_size is not None else "null"

        output['labels'] = {}
        for loc in self.labels:
            output['labels'][f"loc-{loc}"] = list(self.labels[loc])

        with open(savepath, 'w') as f:
            json.dump(output, f)

    @staticmethod
    def load(loadpath):
        with open(loadpath) as f:
            data = json.load(f)

        obstacles = set(map(tuple, data["obstacles"]))
        unknown = set(map(tuple, data["unknown"]))

        labels = {}
        for loc_str in data['labels']:
            loc = eval(loc_str.split('-')[1])
            labels[loc] = set(data['labels'][loc_str])
        return GridMap(data["width"],
                       data["length"],
                       obstacles,
                       unknown=unknown,
                       name=data["name"],
                       ranges_in_metric=data["ranges_in_metric"],
                       grid_size=data["grid_size"],
                       labels=labels)
