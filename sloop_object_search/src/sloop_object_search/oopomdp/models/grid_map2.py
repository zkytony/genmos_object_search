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

    def add_grids(self, obstacles=None, free_locations=None):
        if obstacles is not None:
            self.obstacles |= obstacles
        if free_locations is not None:
            self.free_locations |= free_locations

        # check that obstacles and free locations are disjoint
        overlap = self.obstacles.intersection(self.free_locations)
        if len(overlap) > 0:
            raise ValueError("Error in argument: obstacles and free_locations are not disjoint. "\
                             f"Overlap: {overlap}")

    def all_grids(self):
        return self.free_locations | self.obstacles

    def is_unknown(self, loc):
        return not (loc in self.free_location or loc in self.obstacles)

    @property
    def width(self):
        all_grids = self.all_grids
        min_x = min(all_grids, key=lambda g: g[0])[0]
        max_x = max(all_grids, key=lambda g: g[0])[0]
        return max_x - min_x

    @property
    def length(self):
        all_grids = self.all_grids
        min_y = min(all_grids, key=lambda g: g[1])[1]
        max_y = max(all_grids, key=lambda g: g[1])[1]
        return max_y - min_y

    @property
    def min_corner(self):
        """The (x,y) location with the minimum x, and minimum y;
        This is the corner grid, which, combined with 'width' and
        'length', could be used for iterating over the lattice
        covered by this grid map.
        """
        all_grids = self.all_grids
        min_x = min(all_grids, key=lambda g: g[0])[0]
        min_y = min(all_grids, key=lambda g: g[1])[1]
        return (min_x, min_y)

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
