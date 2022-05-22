import pomdp_py
from ..common import RobotStatus

class RobotStateTopo(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, topo_nid, robot_status=RobotStatus()):
        super().__init__("robot", {"id": robot_id,
                                   "pose": pose,
                                   "topo_nid": topo_nid,
                                   "status": robot_status})

    @property
    def nid(self):
        return self.topo_nid
