














# Experimenting a new way to write POMDPs with pomdp_py
# that is less repetitive.
def static_object_transition(objid, state, action):
    return stat.s(objid).copy()



import pomdp_py

############### State ################
@dataclass(init=True, frozen=True, eq=True, unsafe_hash=True)
class RobotStatus:
    found_objects: Tuple = field(default_factory=lambda: tuple())
    def __str__(self):
        return f"found_objects: {self.found_objects}"

    def copy(self):
        return RobotStatus(self.found_objects)


class RobotStateTopo(pomdp_py.ObjectState):
    def __init__(self,
                 robot_id,
                 pose,
                 topo_nid,
                 robot_status=RobotStatus()):
        super().__init__("robot", {"id": robot_id,
                                   "pose": pose,
                                   "topo_nid": topo_nid,
                                   "status": robot_status})

    @property
    def nid(self):
        return self.topo_nid

class ObjectState(pomdp_py.ObjectState):
    """Object state, specified by object ID, class and its location"""
    def __init__(self, objid, objclass, pose):
        super().__init__(objclass, {"pose": pose, "id": objid})

    def __hash__(self):
        return hash((self.id, self.pose))

    @property
    def pose(self):
        return self['pose']

    @property
    def id(self):
        return self['id']

class JointState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)


############### Action ################
class Motion(pomdp_py.SimpleAction):
    """Motion moves the robot.
    The specific definition is domain-dependent"""

    def __repr__(self):
        return str(self)


class Done(pomdp_py.SimpleAction):
    """Declares the task to be over"""
    def __init__(self):
        super().__init__("done")

    def __repr__(self):
        return str(self)

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


################ Observation ###############
class ObjectObservation(pomdp_py.SimpleObservation):
    def __init__(self, objid, pose):
        self.objid = objid
        self.pose = pose
        super().__init__((objid, pose))

    def __str__(self):
        return f"object({self.objid}, {self.pose})"


class JointObservation(pomdp_py.Observation):
    def __init__(self, objobzs):
        self._hashcache = -1
        self._objobzs = objobzs

    def __hash__(self):
        if self._hashcache == -1:
            self._hashcache = det_dict_hash(self._objobzs)
        return self._hashcache

    def __eq__(self, other):
        return self._objobzs == other._objobzs

    def __str__(self):
        objzstr = ""
        for objid in self._objobzs:
            if self._objobzs[objid].loc is not None:
                objzstr += "{}{}".format(objid, self._objobzs[objid].loc)
        return "JointObservation({})".format(objzstr)

    def __repr__(self):
        return str(self)

    def __len__(self):
        # Only care about object observations here
        return len(self._objobzs)

    def __iter__(self):
        # Only care about object observations here
        return iter(self._objobzs.values())

    def __getitem__(self, objid):
        # objid can be either object id or robot id.
        return self.z(objid)

    def z(self, objid):
        if objid in self._objobzs:
            return self._objobzs[objid]
        else:
            raise ValueError("Object ID {} not in observation".format(objid))


############### Transition Model ##############
class StaticObjectTransitionModel()


############### Agent ################
class TopoSearchAgent(pomdp_py.Agent):
    def __init__(self, robot_id, topo_map, init_belief):
        """
        init_belief (dict): maps from object id to a distribution
            that represents initial belief about the object location.
        """
        self.robot_id = robot_id
        self.topo_map = topo_map


        object_trans_models = {
            objid:
        }
        transition_model = pomdp_py.OOTransitionModel({

        })
        observation_model = TopoSearchTransModel()
