import pomdp_py
from ..common import RobotStatus

class TopoObjectState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, topo_nid):
        super().__init__(self, objclass, {"id": objid,
                                          "topo_nid": topo_nid})

class TopoRobotState(pomdp_py.ObjectState):
    def __init__(self,
                 robot_id,
                 topo_nid,
                 status=RobotStatus()):
        super().__init__("robot", {"id": robot_id,
                                   "topo_nid": topo_nid,
                                   "status": status})

class TopoJointState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)
