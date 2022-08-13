class SearchRegion:
    """model of the search region"""
    def __init__(self):
        pass


class SearchRegion2D:
    """The 2D search region is represented as a GridMap"""
    def __init__(self, grid_map):
        """
        Args:
            grid_map (GridMap):
        """
        pass


class SearchRegion3D:
    """The 3D search region is represented as a collection
    of surfaces, using Open3D's datastructure; This is to
    support reasoning about occlusion while being efficient."""
    def __init__(self):
        pass
