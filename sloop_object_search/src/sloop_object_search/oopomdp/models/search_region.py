from sloop_object_search.utils.conversion import Frame, convert, convert_cov, convert_d
from .octree_belief import OccupancyOctreeDistribution, RegionalOctreeDistribution, DEFAULT_VAL

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
    def is_3d(cls):
        raise NotImplementedError

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
        """converts a pomdp position to a world frame position"""
        return convert(p, Frame.POMDP_SPACE, Frame.WORLD,
                       region_origin=self.region_origin,
                       search_space_resolution=self.search_space_resolution)

    def to_pomdp_pos(self, world_point):
        """converts a pomdp position to a world frame position"""
        return convert(world_point, Frame.WORLD, Frame.POMDP_SPACE,
                       region_origin=self.region_origin,
                       search_space_resolution=self.search_space_resolution)

    def to_region_pos(self, p):
        """converts a pomdp position to a world frame position"""
        return convert(p, Frame.POMDP_SPACE, Frame.REGION,
                       search_space_resolution=self.search_space_resolution)

    def to_world_dpos(self, p):
        """converts a change in pomdp position to a world frame position"""
        return convert_d(p, Frame.POMDP_SPACE, Frame.WORLD,
                         search_space_resolution=self.search_space_resolution)

    def to_pomdp_dpos(self, world_point):
        """converts a change in pomdp position to a world frame position"""
        return convert_d(world_point, Frame.WORLD, Frame.POMDP_SPACE,
                         search_space_resolution=self.search_space_resolution)

    def to_region_dpos(self, p):
        """converts a change in pomdp position to a world frame position"""
        return convert_d(p, Frame.POMDP_SPACE, Frame.REGION,
                         search_space_resolution=self.search_space_resolution)


    def to_world_cov(self, cov):
        """given a covariance for the pomdp frame to the world frame"""
        return convert_cov(cov, Frame.POMDP_SPACE, Frame.WORLD,
                           search_space_resolution=self.search_space_resolution,
                           is_3d=self.is_3d)

    def to_pomdp_cov(self, world_cov):
        """given a covariance for the world frame to the pomdp frame"""
        return convert_cov(world_cov, Frame.WORLD, Frame.POMDP_SPACE,
                           search_space_resolution=self.search_space_resolution,
                           is_3d=self.is_3d)

    def to_region_cov(self, cov):
        """given a covariance for the POMDP frame to the region frame"""
        return convert_cov(cov, Frame.POMDP_SPACE, Frame.REGION,
                           search_space_resolution=self.search_space_resolution,
                           is_3d=self.is_3d)

    def to_world_pose(self, pose, cov=None):
        """Given a pose in POMDP frame, return a pose in the WORLD frame."""
        if self.is_3d:
            pos_len = 3
        else:
            pos_len = 2
        pos_world = self.to_world_pos(pose[:pos_len])
        rot_world = pose[pos_len:]
        pose_world = (*pos_world, *rot_world)
        if cov is not None:
            cov_world = self.to_world_cov(cov)
            return pose_world, cov_world
        else:
            return pose_world

    def to_region_pose(self, pose, cov=None):
        """Given a pose in REGION frame, return a pose in the POMDP frame."""
        if self.is_3d:
            pos_len = 3
        else:
            pos_len = 2
        pos_region = self.to_region_pos(pose[:pos_len])
        rot_region = pose[pos_len:]
        pose_region = (*pos_region, *rot_region)
        if cov is not None:
            cov_region = self.to_region_cov(cov)
            return pose_region, cov_region
        else:
            return pose_region

    def to_pomdp_pose(self, world_pose, cov=None):
        """Given a pose in WORLD frame, return a pose in the POMDP frame."""
        if self.is_3d:
            pos_len = 3
        else:
            pos_len = 2
        pos_pomdp = self.to_pomdp_pos(world_pose[:pos_len])
        rot_pomdp = world_pose[pos_len:]
        pose_pomdp = (*pos_pomdp, *rot_pomdp)
        if cov is not None:
            cov_pomdp = self.to_pomdp_cov(cov)
            return pose_pomdp, cov_pomdp
        else:
            return pose_pomdp

    def __len__(self):
        # size of the search region
        raise NotImplementedError()


class SearchRegion2D(SearchRegion):
    """The 2D search region is represented as a GridMap.
    For now, the free locations on the grid map are potential
    object locations. TODO: change this to be label-based?"""
    def __init__(self, grid_map, region_origin=None, grid_size=None):
        """
        Args:
            grid_map (GridMap2):
        """
        super().__init__(grid_map,
                         region_origin=region_origin,
                         search_space_resolution=grid_size)

    @property
    def is_3d(cls):
        return False

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

    def __contains__(self, pos):
        return pos in self.grid_map.free_locations




class SearchRegion3D(SearchRegion):
    """The 3D search region is represented as an octree_dist
    We anticipate a 3D search region is necessary only for
    local search.

    Note that the default value of a node in this octree
    should be zero, indicating free space (it is not a belief,
    it is occupancy)."""
    def __init__(self, octree_dist, **kwargs):
        """octree_dist should be an OccupancyOctreeDistribution"""
        assert isinstance(octree_dist, OccupancyOctreeDistribution),\
            "octree_dist must be OccupancyOctreeDistribution."
        super().__init__(octree_dist, **kwargs)

    @property
    def is_3d(cls):
        return True

    @property
    def octree_dist(self):
        return self._region_repr

    @octree_dist.setter
    def octree_dist(self, octree_dist):
        self._region_repr = octree_dist

    def to_octree_pos(self, world_point):
        return self.to_pomdp_pos(world_point)

    def occupied_at(self, pos, res=1):
        if not self.octree_dist.octree.valid_voxel(*pos, res):
            raise ValueError(f"position {pos} at resolution {res} is not a valid voxel.")
        node = self.octree_dist.octree.get_node(*pos, res)
        if node is None:
            return False
        else:
            return node.value() > 0

    def valid_voxel(self, voxel):
        """a voxel is a tuple (x, y, z, res)"""
        return self.octree_dist.octree.valid_voxel(*voxel)

    def project_to_2d(self, pos, search_region2d):
        assert len(pos) == 3, "pos must be 3D"
        world_pos = self.to_world_pos(pos)
        return search_region2d.to_pomdp_pos(world_pos[:2])


class LocalRegionalOctreeDistribution(RegionalOctreeDistribution):
    """This is specifically for local search agent in hierarchical planning. The
    region's height is given by a height limit and its footprint is determined
    by a 2D search region; This ensures the local octree belief fits within
    the 2D search region."""
    def __init__(self, local_search_region, global_search_region,
                 default_region_val=DEFAULT_VAL, **kwargs):
        assert isinstance(local_search_region, SearchRegion3D)
        assert isinstance(global_search_region, SearchRegion2D)
        dimensions = local_search_region.octree_dist.octree.dimensions
        self._local_search_region = local_search_region
        self._global_search_region = global_search_region
        super().__init__(dimensions, region=local_search_region.octree_dist.region,
                         default_region_val=default_region_val, **kwargs)

    def in_region(self, voxel):
        if not super().in_region(voxel):
            return False
        else:
            # check if the voxel is within the 2D map in global search region
            x, y, z, r = voxel
            # check center and corners
            voxel_center = (x*r + r/2, y*r + r/2, z*r + r/2)
            voxel_min_origin = (x*r, y*r, z*r)
            voxel_max_origin = (x*r + r, y*r + r, z*r + r)

            voxel_center_2d = self._local_search_region.project_to_2d(voxel_center, self._global_search_region)
            if voxel_center_2d not in self._global_search_region:
                return False

            voxel_min_origin_2d = self._local_search_region.project_to_2d(voxel_min_origin, self._global_search_region)
            if voxel_min_origin_2d not in self._global_search_region:
                return False

            voxel_max_origin_2d = self._local_search_region.project_to_2d(voxel_max_origin, self._global_search_region)
            if voxel_max_origin_2d not in self._global_search_region:
                return False
            return True

    def sample_from_region(self):
        # avoid sampling from outside the global region
        xr, yr, zr = super().sample_from_region()
        while not self.in_region((xr, yr, zr, 1)):
            xr, yr, zr = super().sample_from_region()
        return (xr, yr, zr)
