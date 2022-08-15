class SearchRegion:
    """model of the search region"""
    def __init__(self):
        pass


class SearchRegion2D(SearchRegion):
    """The 2D search region is represented as a GridMap"""
    def __init__(self, grid_map):
        """
        Args:
            grid_map (GridMap2):
        """
        self.grid_map = grid_map


class SearchRegion3D(SearchRegion):
    """The 3D search region is represented as an octree.
    We anticipate a 3D search region is necessary only for
    local search.

    Note that the default value of a node in this octree
    should be zero (it is not a belief, it is occupancy)."""
    def __init__(self, octree):
        self.octree = octree
