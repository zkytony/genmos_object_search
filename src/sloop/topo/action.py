from ..common import Motion

class MoveTopo(Motion):
    def __init__(self, src_nid, dst_nid, gdist=None,
                 cost_scaling_factor=1.0, atype="move"):
        """
        Moves the robot from src node to dst node
        """
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        self.gdist = gdist
        self._cost_scaling_factor = cost_scaling_factor
        super().__init__("{}({}->{})".format(atype, self.src_nid, self.dst_nid))

    @property
    def step_cost(self):
        return min(-(self.gdist * self._cost_scaling_factor), -1)


class Stay(MoveTopo):
    """the stay action is basically MoveTopo but actually not changing the node"""
    def __init__(self, nid, cost_scaling_factor=1.0):
        super().__init__(nid, nid, gdist=0.0, cost_scaling_factor=1.0, atype="stay")
