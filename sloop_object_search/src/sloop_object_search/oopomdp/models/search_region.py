from sloop_object_search.utils.conversion import Frame, convert

class SearchRegion:
    """model of the search region. The idea of search region is
    it represents the set of possible object locations and it
    also defines unreachable positions by the robot (because of
    obstacles)."""
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

    def to_world_pos(self, p):
        """converts a grid position to a world frame position"""
        return convert(p, Frame.POMDP_SPACE, Frame.WORLD,
                       region_origin=self.region_origin,
                       search_space_resolution=self.search_space_resolution)

    def to_pomdp_pos(self, world_point):
        """converts a grid position to a world frame position"""
        return convert(world_point, Frame.WORLD, Frame.POMDP_SPACE,
                       region_origin=self.region_origin,
                       search_space_resolution=self.search_space_resolution)

    def to_region_pos(self, p):
        """converts a grid position to a world frame position"""
        return convert(p, Frame.POMDP_SPACE, Frame.REGION,
                       search_space_resolution=self.search_space_resolution)

    def __len__(self):
        # size of the search region
        raise NotImplementedError()


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

    def to_grid_pos(self, world_point):
        return self.to_pomdp_pos(world_point)

    def __iter__(self):
        return iter(self.grid_map.free_locations)

    def __len__(self):
        return len(self.grid_map.free_locations)


class SearchRegion3D(SearchRegion):
    """The 3D search region is represented as an octree_dist
    (OctreeDistribution).
    We anticipate a 3D search region is necessary only for
    local search.

    Note that the default value of a node in this octree
    should be zero, indicating free space (it is not a belief,
    it is occupancy)."""
    @property
    def octree_dist(self):
        return self._region_repr

    @octree_dist.setter
    def octree_dist(self, octree_dist):
        self._region_repr = octree_dist

    def to_octree_pos(self, world_point):
        return self.to_pomdp_pos(world_point)

    def occupied_at(self, pos, res=1):
        if not self.octree.valid_voxel(*pos, res):
            raise ValueError(f"position {pos} at resolution {res} is not a valid voxel.")
        node = self.octree_dist.octree.get_node(*pos, res)
        if node is None:
            return False
        else:
            return node.value() > 0

    def valid_voxel(self, voxel):
        """a voxel is a tuple (x, y, z, res)"""
        return self.octree_dist.octree.valid_voxel(*voxel)
