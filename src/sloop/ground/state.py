import pomdp_py

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
