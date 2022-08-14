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
# are integers, and there will always be a (0,0) grid.
#
# One benefit of not requiring non-negative coordinates is that the
# grid map can be extended in any direction, while keeping the coordinates
# of known grid cells unchanged.

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
