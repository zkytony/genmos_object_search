"""Egocentric representation of objects (provides a generic
interface); Specifically considers graph-based ego-centric
representation."""
from .graph import Node, Edge, Graph
from . import util
import copy

class EgoRep:
    def __init__(self, name):
        """An EgoRep (egocentric spatial representation) is created by
        specifying the `name` of the ego, i.e. the thing of concern
        (e.g. an object)."""
        self._name = name

    @property
    def name(self):
        return self._name

    def transform(self, world_point):
        """converts a world point into the coordinate
        system of the egocentric representation"""
        raise NotImplementedError


class ViewNode(Node):
    """An ViewNode is a node that stores the likelihoods of object existance
    at a specific view. This node doesn't have a actual coordinate (it's relative
    position is defined by an EgoGraph)."""
    def __init__(self, id, viewnum, lh={}, data=None):
        """
        lh (dict): map from object (an integer, same id in agent belief)
            to a float indicating the likelihood.
        """
        # make sure all likelihoods are not negative
        for obj in lh:
            if type(lh[obj]) != float\
               or lh[obj] < 0.0:
                raise ValueError("Invalid likelihood dictionary at %s: %s"
                                 % (str(obj), str(lh[obj])))
        self._lh = lh
        self._viewnum = viewnum
        self.id = id
        self.data = data

    @property
    def viewnum(self):
        return self._viewnum

    def lh_obj(self, obj):
        if obj in self._lh:
            return self._lh[obj]
        else:
            return 0.0

    def set_lh(self, obj, val):
        self._lh[obj] = val

    @property
    def size(self):
        return self.data

    @size.setter
    def size(self, value):
        self.data = value


class ViewEdge(Edge):
    @property
    def length(self):
        return self.data

    @length.setter
    def length(self, value):
        self.data = length


class CenterNode(Node):
    """A CenterNode is the center node in the egocentric graph
    representation. It has a name."""
    def __init__(self, id, name):#, coords=None):
        self._name = name
        self.id = id

    @property
    def name(self):
        return self._name


class EgoGraph(EgoRep, Graph):
    def __init__(self,
                 name,
                 edges=None,
                 divisions=8,
                 objects=set({})):
        """Divides up the 360 degree centered at the object with `name`
        into `divisions` number of fan-shaped divisions (called view). Each
        division has a number (index), starting from 0 to divisions-1.
        There is an EgoNode connected through each view.

        Args:
            name (str): name of the object at center
            edges (dict or set): ViewEdges in this graph (It's optional).
            objects (set): set of objects whose presence is considered.
                By default, the prob.of.existance of all objects is 50% at each
                ViewNode on the EgoGraph. The objects should be represented
                by their ids in the oopomdp.
        """
        self._center_node = CenterNode(0, name)
        self._objects = objects
        self._divisions = divisions
        if edges is None:
            edges = {}  # map from edge id (viewnum) to Edge
            for viewnum in range(divisions):
                view_node = ViewNode(self.vnid(viewnum),
                                     viewnum, lh={obj:1e-61  # default likelihood ~0%
                                                  for obj in objects},
                                     data=1)  # data means size
                edge = ViewEdge(viewnum, self._center_node,
                                view_node, data=1)  # data: length of the edge
                edges[edge.id] = edge
        EgoRep.__init__(self, name)
        Graph.__init__(self, edges)

    def vnid(self, viewnum):
        """node id for view node"""
        return 10 + viewnum

    def node_at(self, viewnum):
        """viewnum (int) the view index; If -1, then it refers to the center."""
        if viewnum < 0:
            return self._center_node
        else:
            if self.vnid(viewnum) in self.nodes:
                return self.nodes[self.vnid(viewnum)]
            else:
                raise ValueError("viewnum %d does not match any node" % viewnum)

    def lh_obj(self, viewnum, objid):
        return self.node_at(viewnum).lh_obj(objid)

    def edge_at(self, viewnum):
        if viewnum in self.edges:
            return self.edges[viewnum]
        else:
            raise ValueError("viewnum %d does not match any edge" % viewnum)

    @property
    def center_node(self):
        return self._center_node

    @property
    def divisions(self):
        return self._divisions

    @property
    def objects(self):
        """Objects whose likelihood of presence is modeled"""
        return self._objects

    def __repr__(self):
        return "EgoGraph(%s,%d)" % (self.name, self.divisions)


class GroundedEgoGraph(EgoGraph):

    def __init__(self, egograph, groundings):
        """
        egograph (EgoGraph): an EgoGraph that we are grounding
        groundings (dict): a dictionary mapping each node in
            the graph (by id) to specific coordinates (in a grid map).
        """
        for nid in egograph.nodes:
            if nid not in groundings:
                raise ValueError("Node %d in the EgoGraph is not grounded." % nid)
        self._groundings = groundings
        super().__init__(egograph.name,
                         edges=copy.deepcopy(egograph.edges),
                         divisions=egograph.divisions,
                         objects=egograph.objects)

    def coords_by_viewnum(self, viewnum):
        node = self.node_at(viewnum)
        return self._groundings[node.id]

    def coords_by_id(self, nid):
        return self._groundings[nid]

    def coords(self, i, by_id=False):
        """
        Returns the grounded coordinates of i; These coords MAY NOT be integers.
            i could be either: a node id, a viewnum, or -1, referring to the center.
            Priority is given in this order.
        """
        if by_id:
            return self.coords_by_id(i)
        else:
            return self.coords_by_viewnum(i)

    @classmethod
    def ground(self, egograph, params, grid_map, warning_only=True):
        """
        Grounds the egograph onto the grid_map, with
        specified parameters (i.e. distance to
        the center in each direction, or something else).

        TODO: What parameters do we need? For now:
        params is a dictionary mapping from viewnum to distances,
            and the value -1 (representing center) to a location,
            and contains phase shift info.
        """
        assert -1 in params,\
            "The coordinates of egograph center must be given."

        phase_shift = 0
        if "phase_shift" in params:
            phase_shift = params["phase_shift"]

        groundings = {egograph.center_node.id: params[-1]}
        for viewnum in range(egograph.divisions):
            assert viewnum in params,\
                "The distance to center for viewnum %d must be given." % viewnum
            dist = params[viewnum]
            coords = \
                util.compute_view_node_coords(
                    groundings[egograph.center_node.id],  # center coords
                    viewnum, dist,
                    divisions=egograph.divisions,
                    phase_shift=phase_shift)
            groundings[egograph.node_at(viewnum).id] = coords
        # verify
        for nid in groundings:
            if not grid_map.within_range(*groundings[nid]):
                if warning_only:
                    print("Warning: Grounding of node %d is not on the grid map." % nid)
                else:
                    assert grid_map.within_range(groundings[nid]),\
                        "Grounding of node %d is not on the grid map." % nid
        return GroundedEgoGraph(egograph, groundings)

    def __repr__(self):
        return "GroundedEgoGraph(%s,%d @ %s)"\
            % (self.name, self.divisions, str(self.coords_by_viewnum(-1)))
