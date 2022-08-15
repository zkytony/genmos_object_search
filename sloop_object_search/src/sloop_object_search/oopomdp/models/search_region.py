from sloop_object_search.utils.conversion import Frame, convert

class SearchRegion:
    """model of the search region"""
    def __init__(self, region_repr,
                 region_origin=None, search_space_resolution=None):
        """
        region_repr (object): representation of the region
        region_origin (tuple): The world frame coordinates (metric) of
            the (0,0) grid cell. Used for conversion between grid and world
            coordinates.
        search_space_resolution (float): the metric resolution of one
         coordinate length in the POMDP space (e.g. 0.15 means one
         POMDP grid equals to 0.15m in the real world)
        """
        self._region_repr = region_repr
        self._region_origin = region_origin
        self._search_space_resolution = search_space_resolution

    @property
    def region_repr(self):
        return self._region_repr

    @property
    def region_origin(self):
        return self._region_origin

    @property
    def search_space_resolution(self):
        return self._search_space_resolution

    def to_world_pos(self, x, y):
        """converts a grid position to a world frame position"""
        return convert((x, y), Frame.POMDP_SPACE, Frame.WORLD,
                       region_origin=self.region_origin,
                       search_space_resolution=self.search_space_resolution)

    def to_grid_pos(self, world_x, world_y):
        """converts a grid position to a world frame position"""
        return convert((world_x, world_y), Frame.WORLD, Frame.POMDP_SPACE,
                       region_origin=self.region_origin,
                       search_space_resolution=self.search_space_resolution)

    def to_region_pos(self, x, y):
        """converts a grid position to a world frame position"""
        return convert((x, y), Frame.POMDP_SPACE, Frame.REGION,
                       search_space_resolution=self.search_space_resolution)


class SearchRegion2D(SearchRegion):
    """The 2D search region is represented as a GridMap"""
    def __init__(self, grid_map, region_origin=None, grid_size=None):
        """
        Args:
            grid_map (GridMap2):
        """
        super().__init__(grid_map,
                         region_origin=region_origin,
                         search_space_resolution=grid_size)

    @property
    def grid_map(self):
        return self.region_repr

    @property
    def grid_size(self):
        return self.search_space_resolution


class SearchRegion3D(SearchRegion):
    """The 3D search region is represented as an octree.
    We anticipate a 3D search region is necessary only for
    local search.

    Note that the default value of a node in this octree
    should be zero (it is not a belief, it is occupancy)."""
    def __init__(self, octree, region_origin=None):
        self.octree = octree
        self.region_origin = region_origin
