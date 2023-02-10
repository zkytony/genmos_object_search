from genmos_object_search.oopomdp.domain.state import RobotState

class RobotStateTopo(RobotState):
    """Represents a robot state on a topological graph. Note
    that when comparing two RobotStateTopo objects, we ignore
    their pose but pay attention to their topo_nid and topo_hashcode."""
    def __init__(self,
                 robot_id,
                 pose,
                 objects_found,
                 camera_direction,
                 topo_nid):
        super().__init__(robot_id,
                         pose,
                         objects_found,
                         camera_direction,
                         topo_nid=topo_nid,
                         topo_map_hashcode=None)   # to make system test pass

    @property
    def nid(self):
        return self['topo_nid']

    @property
    def topo_nid(self):
        return self['topo_nid']

    def __str__(self):
        return "{}({}, {}, nid={})".format(type(self).__name__, self.pose, self.objects_found, self.topo_nid)
