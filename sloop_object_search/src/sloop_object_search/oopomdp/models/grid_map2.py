# An improvement over GridMap.
# Not changing GridMap directly because much code depends on it
# (especially for simulation)
#
# The biggest differences GridMap2 has over GridMap are:
#
# - Grid coordinates don't have to be non-negative.
# - There is no width and length as an input parameter to build a GridMap2;
#   they are inferred based on the given grids.
# - The conversion between grid coordinates and metric coordinates
#   are done differently; simpler and more cleanly.
#
# GridMap2 will emphasize the idea that a grid map is a collection
# of grids. Indeed, these grids live on a lattice, but the construction
# of a GridMap2 does not require specifying every grid on the lattice.
# Unspecified grids will have "unknown" occupancy.
#
# Note that because the grids live on a lattice, the grid coordinates
# are integers, and there will always be a (0,0) grid. This forms the
# coordinate system of the 2D object search agent.
#
# One benefit of not requiring non-negative coordinates is that the
# grid map can be extended in any direction, while keeping the coordinates
# of known grid cells unchanged.

from sloop_object_search.utils.conversion import Frame, convert
from .grid_map import GridMap

class GridType:
    FREE = "free_location"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"


class GridMap2:
    def __init__(self, name="grid_map2",
                 obstacles=None, free_locations=None,
                 world_origin=None, grid_size=None, labels=None):
        """
        Args:
            name (str): name of the grid map
            obstacles (set): a set of (x,y) locations for the obstacles
            free_locations (set): a set of (x,y) locations for the free locations
            world_origin (tuple): The world frame coordinates (metric) of
                the (0,0) grid cell. Used for conversion between grid and world
                coordinates.
            grid_size (float): The metric size of a grid. For example, 0.25 means
                each grid cell has length equal to 0.25m in the world frame.
            labels (dict): maps from location to a set of string labels
        """
        if obstacles is None and free_locations is None:
            raise ValueError("Need to supply obstacles or free_locations (or both)")

        self.obstacles = set()
        self.free_locations = set()
        self.add_grids(obstacles=obstacles, free_locations=free_locations)

        self.world_origin = world_origin
        self.grid_size = grid_size

        self.labels = labels
        if labels is None:
            self.labels = {}  # maps from position (x,y) to a set of labels

        # caches
        self._width_cache = None
        self._length_cache = None
        self._min_corner_cache = None


    def add_grids(self, obstacles=None, free_locations=None):
        """note: user should not modify the content of self.obstacles
        and self.free_locations, without calling add_grids"""
        if obstacles is not None:
            self.obstacles |= obstacles
        if free_locations is not None:
            self.free_locations |= free_locations
        self.check_integrity()
        self._clear_cache()


    def update_region(self, obstacles=None, free_locations=None):
        """Update the rectangular region covered by 'obstacles' and free_locations: the
        obstacles and free_locations in that region are replaced by the given
        'obstacles' and 'free_locations'.
        """
        # First, we create a temporary GridMap2 with the given
        region_grid_map = GridMap2(name="tmp", obstacles=obstacles,
                                   free_locations=free_locations)
        # Enumerate over the locations in this region and update
        for x in region_grid_map.wrange:
            for y in region_grid_map.lrange:
                # make the type of (x,y) in 'self' match with region_grid_map
                current_grid_type = self.grid_type((x,y))
                new_grid_type = region_grid_map.grid_type((x,y))
                if current_grid_type == new_grid_type:
                    continue
                else:
                    if new_grid_type == GridType.FREE:
                        self.free_locations.add((x,y))
                        if current_grid_type == GridType.OBSTACLE:
                            self.obstacles.remove((x,y))
                        else:
                            assert current_grid_type == GridType.UNKNOWN
                            # nothing to do.
                    elif new_grid_type == GridType.OBSTACLE:
                        self.obstacles.add((x,y))
                        if current_grid_type == GridType.FREE:
                            self.free_locations.remove((x,y))
                        else:
                            assert current_grid_type == GridType.UNKNOWN
                            # nothing to do.
                    else:
                        assert new_grid_type == GridType.UNKNOWN
                        if current_grid_type == GridType.FREE:
                            self.free_locations.remove((x,y))
                        else:
                            assert current_grid_type == GridType.OBSTACLE
                            self.obstacles.remove((x,y))
        self.check_integrity()
        self._clear_cache()


    def check_integrity(self):
        # check that obstacles and free locations are disjoint
        overlap = self.obstacles.intersection(self.free_locations)
        if len(overlap) > 0:
            raise ValueError("Error in argument: obstacles and free_locations are not disjoint. "\
                             f"Overlap: {overlap}")

    @property
    def all_grids(self):
        return self.free_locations | self.obstacles

    def grid_type(self, loc):
        if loc in self.free_locations:
            return GridType.FREE
        elif loc in self.obstacles:
            return GridType.OBSTACLE
        return GridType.UNKNOWN

    def _clear_cache(self):
        self._width_cache = None
        self._length_cache = None
        self._min_corner_cache = None

    @property
    def width(self):
        if self._width_cache is not None:
            return self._width_cache
        all_grids = self.all_grids
        min_x = min(all_grids, key=lambda g: g[0])[0]
        max_x = max(all_grids, key=lambda g: g[0])[0]
        w = max_x - min_x
        self._width_cache = w
        return w

    @property
    def length(self):
        if self._length_cache is not None:
            return self._length_cache
        all_grids = self.all_grids
        min_y = min(all_grids, key=lambda g: g[1])[1]
        max_y = max(all_grids, key=lambda g: g[1])[1]
        l = max_y - min_y
        self._length_cache = l
        return l

    @property
    def lrange(self):
        """a 'range' object over y coordinates in this grid map"""
        return range(self.min_corner[1], self.min_corner[1] + self.length)

    @property
    def wrange(self):
        """a 'range' object over x coordinates in this grid map"""
        return range(self.min_corner[0], self.min_corner[0] + self.width)

    @property
    def min_corner(self):
        """The (x,y) location with the minimum x, and minimum y;
        This is the corner grid, which, combined with 'width' and
        'length', could be used for iterating over the lattice
        covered by this grid map.
        """
        if self._min_corner_cache is not None:
            return self._min_corner_cache
        all_grids = self.all_grids
        min_x = min(all_grids, key=lambda g: g[0])[0]
        min_y = min(all_grids, key=lambda g: g[1])[1]
        c = (min_x, min_y)
        self._min_corner_cache = c
        return c

    def to_world_pos(self, x, y):
        """converts a grid position to a world frame position"""
        return convert((x, y), Frame.POMDP_SPACE, Frame.WORLD,
                       region_origin=self.world_origin,
                       search_space_resolution=self.grid_size)

    def to_grid_pos(self, world_x, world_y):
        """converts a grid position to a world frame position"""
        return convert((world_x, world_y), Frame.WORLD, Frame.POMDP_SPACE,
                       region_origin=self.world_origin,
                       search_space_resolution=self.grid_size)

    def to_grid_map(self):
        """Converts a GridMap2 to a GridMap. This is convenient
        for visualization purpose, because our visualization
        code is based on GridMap."""
        # need to shift all the free locations and obstacles
        shifted_obstacles = set(self.shift_pos(*loc) for loc in self.obstacles)
        shifted_free_locations = set(self.shift_pos(*loc) for loc in self.free_locations)
        ranges_in_metric = None
        if self.world_origin is not None\
           and self.grid_size is not None:
            ranges_in_metric = [
                (self.world_origin[0], self.world_origin[0] + self.width*self.grid_size),
                (self.world_origin[1], self.world_origin[1] + self.length*self.grid_size)
            ]
        shifted_labels = {}
        for loc in self.labels:
            shifted_labels[self.shift_pos(*loc)] = self.labels[loc]
        return GridMap(self.width, self.length,
                       shifted_obstacles,
                       free_locations=shifted_free_locations,
                       ranges_in_metric=ranges_in_metric,
                       grid_size=self.grid_size,
                       labels=self.labels)

    def shift_pos(self, x, y):
        """Given a position (x,y) on GridMap2, return
        a shifted position relative to 'min_corner'; The
        resulting position has guaranteed non-negative
        coordinates. """
        min_corner = self.min_corner
        return (x - min_corner[0], y - min_corner[1])

    def shift_back_pos(self, x, y):
        """
        Given (x,y) that is nonnegative, i.e. with respect
        to a (0,0)-corner, return a corresponding position
        on GridMap2
        """
        min_corner = self.min_corner
        return (x + min_corner[0], y + min_corner[1])
