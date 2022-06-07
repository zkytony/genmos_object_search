### CODE copied from zkytony/graphspn repository.

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
import copy
from collections import deque

########################################
#  Node
########################################
class Node:

    def __init__(self, node_id, data=None):
        """
        The id is expected to be unique in the graph.
        """
        self.id = node_id
        self.data = data

    @property
    def coords(self):
        """returns an (x,y) location on the plane, for visualization purposes.
        The coordinates have resolution """
        if hasattr(self, "_coords"):
            return self._coords
        else:
            self._coords = (random.randint(-500, 500),
                            random.randint(-500, 500))
            return self._coords

    def __repr__(self):
        return "%s(%d)" % (type(self).__name__, self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id\
                and self.data == other.data
        return False

class SuperNode(Node, ABC):

    def __init__(self, id, enset):
        """
        id (int) is for this node
        enset (OrderedEdgeNodeSet) underlying graph structure
        """
        self.id = id
        if type(enset) != OrderedEdgeNodeSet:
            raise ValueError("super node must use OrderedEdgeNodeSet!")
        self.enset = enset

    @property
    def data(self):
        return self.enset

    @classmethod
    @abstractmethod
    def pick_id(cls, enset, existing_ids):
        """
        that always returns the same id if given the same subgraph, and different
        if otherwise.
        """
        pass

    def nodes_list(self):
        return self.enset.nodes_list

########################################
#  Edge
########################################
class Edge:
    """
    An edge links two nodes.
    """

    def __init__(self, eid, node1, node2, data=None):
        """
        The id is expected to be unique in the graph edges.
        """
        self.id = eid
        if node2 is None:
            self.nodes = (node1,)
        else:
            self.nodes = (node1, node2)

        self.data = data

    @property
    def attrs(self):
        return {"data": self.data}

    @property
    def degenerate(self):
        return len(self.nodes) == 1

    def __repr__(self):
        if self.data is None:
            data = "--"
        else:
            data = self.data
        if not self.degenerate:
            return "#%d[<%d>%s<%d>]" % (self.id, self.nodes[0].id, str(data), self.nodes[1].id)
        else:
            return "#%d[<%d>]" % (self.id, self.nodes[0].id)

    def other(self, this_nid):
        if this_nid == self.nodes[0].id:
            return self.nodes[1]
        elif this_nid == self.nodes[1].id:
            return self.nodes[0]
        else:
            raise ValueError("Invalid query {} for other node in edge".format(this_nid))


class SuperEdge(Edge, ABC):

    def __init__(self, id, supernode1, supernode2, data=None):
        """
        id (int) id for this edge
        supernode1 (SuperNode) a super node
        supernode2 (SuperNode) a super node
        """
        super().__init__(id, supernode1, supernode2, data=data)

    @classmethod
    @abstractmethod
    def pick_id(self, supernode1, supernode2, edge=None, existing_ids=set({})):
        """
        function that always returns the same id if given the
        same subgraphs on two ends and an edge on the original graph,
        and different if otherwise.
        """
        pass



class EdgeNodeSet:
    """
    EdgeNodeSet is simply a tuple (V, E) but without restriction on whether the
    edges and vertices have any correspondence. It may be a graph, or not.
    """
    def __init__(self, nodes, edges):
        """
        edges (set or dict) map from edge id to an edge object
        nodes (set or dict) map from node id to a node object

        Both could be None.
        """
        if nodes is None:
            nodes = {}
        if edges is None:
            edges = {}
        if type(nodes) == set:
            nodes = {e.id:e for e in nodes}
        if type(edges) == set:
            edges = {e.id:e for e in edges}

        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def num_nodes(self):
        return len(self._nodes)

    def num_edges(self):
        return len(self._edges)

    def _build_degen_edges(self):
        # Given that edges is empty, returns a set of degenerate edges, each
        # attached to a node
        assert len(self._edges) == 0
        edges = set({})
        for nid in self._nodes:
            edges.add(Edge(nid, self._nodes[nid]))
        return edges

    def to_graph(self, directed=False):
        # Verify integrity
        for eid in self._edges:
            nodes = self._edges[eid].nodes
            if nodes[0].id not in self._nodes:
                return None
            if nodes[1].id not in self._nodes:
                return None
        return Graph(self._edges, directed=directed)

    def to_unused_graph(self, directed=False, **params):
        edges = self._edges
        return UnusedGraph(edges, **params)

# absent_node_ids (set): nodes in this set are actually "not present", meaning
#     they provide the structure for this graph, but they are excluded
#     in function calls, such as neighbors() and partition().

class OrderedEdgeNodeSet(EdgeNodeSet):

    def __init__(self, nodes, edges):
        """
        edges (list) list of edge objects
        nodes (list) list of node objects

        Both could be None.
        """
        super().__init__(set(nodes), set(edges))
        self._ordered_nodes = nodes
        self._ordered_edges = edges

    @property
    def nodes_list(self):
        return self._ordered_nodes

    @property
    def edges_list(self):
        return self._ordered_edges


########################################
#  Graph
########################################
class Graph(EdgeNodeSet):

    def __init__(self, edges, directed=False):
        """
        edges (set or dict) map from edge id to an edge object
        A graph could be a simple/multi-graph and a (un)directed graph.
        """
        # Build nodes map
        nodes = {}
        if type(edges) == set:
            for edge in edges:
                endpoints = edge.nodes
                if endpoints[0].id not in nodes:
                    nodes[endpoints[0].id] = endpoints[0]
                if edge.degenerate: # degenerate edge; only one node.
                    continue
                if endpoints[1].id not in nodes:
                    nodes[endpoints[1].id] = endpoints[1]
            edges = {e.id:e for e in edges}
        elif type(edges) == dict:
            for eid in edges:
                endpoints = edges[eid].nodes
                if endpoints[0].id not in nodes:
                    nodes[endpoints[0].id] = endpoints[0]
                if edges[eid].degenerate:  # degenerate edge; only one node.
                    continue
                if endpoints[1].id not in nodes:
                    nodes[endpoints[1].id] = endpoints[1]

        super().__init__(nodes, edges)
        self._directed = directed

        # keep track of connections for faster graph operations
        self._conns = {nid : {} for nid in self._nodes}
        self._outedges = {nid : set({}) for nid in self._nodes}
        self._build_connections()


    @property
    def directed(self):
        return self._directed

    def is_empty(self):
        return self.num_edges() == 0

    # @abstractmethod
    # def _build_connections(self):
    def _build_connections(self):
        """Builds the self._conns field, which is a map from node id to a dictionary of neighbor id -> edge(s).
        Implementation differs between different types of graphs."""
        self._multi = False
        for eid in self._edges:
            if self._edges[eid].degenerate:  # degenerate, only one node.
                node1 = self._edges[eid].nodes[0]
            else:
                node1, node2 = self._edges[eid].nodes
            if node1.id not in self._conns:
                self._conns[node1.id] = {}
                self._outedges[node1.id] = set({})
            if self._edges[eid].degenerate:  # degenerate, only one node.
                continue
            if node2.id not in self._conns[node1.id]:
                self._conns[node1.id][node2.id] = set({})
            self._conns[node1.id][node2.id].add(eid)
            self._outedges[node1.id].add(eid)
            if len(self._conns[node1.id][node2.id]) > 1:
                self._multi = True
            if not self._directed:
                if node2.id not in self._conns:
                    self._conns[node2.id] = {}
                if node1.id not in self._conns[node2.id]:
                    self._conns[node2.id][node1.id] = set({})
                self._conns[node2.id][node1.id].add(eid)
                self._outedges[node2.id].add(eid)

    #--- Basic graph operations ---#
    def is_neighbor(self, node_id, test_id):
        return test_id in self._conns[node_id]

    def neighbors(self, node_id):
        """
        Returns a set of neighbor node ids
        """
        return set(self._conns[node_id].keys())

    def edges_between(self, node1_id, node2_id):
        """Return edge id(s) between node 1 and node 2; The returned object depends on
        the child class's implementation of _build_connections()"""
        if node2_id not in self._conns[node1_id]:
            return None
        else:
            return self._conns[node1_id][node2_id]# {self.edges[eid] for eid in }

    def edges_from(self, node_id):
        """Return edge id(s) from node_id"""
        return self._outedges[node_id]


    def copy(self):
        """
        Returns a new Graph which contains the same information as `self`. The new object
        is completely separate from `self`, meaning modifying any information in the copied topo-map
        does not affect `self`.
        """
        edges_copy = copy.deepcopy(self._edges)
        return self.__class__(edges_copy)

    def connected_components(self):
        """
        Returns the connected components in this graph, each as a separate TopologicalMap instance.

        Note: The union of the sets of node ids in the returned connected components equals
        to the original topo map's set of node ids. (i.e. node ids are kept the same in components)
        """
        # Uses BFS to find connected components
        copy_graph = self.copy()
        to_cover = set(copy_graph.nodes.keys())
        components = []
        while len(to_cover) > 0:
            start_nid = random.sample(to_cover, 1)[0]
            q = deque()
            q.append(start_nid)
            component_edge_ids = set({})
            visited = set()
            while len(q) > 0:
                nid = q.popleft()
                neighbors = copy_graph.neighbors(nid)
                for neighbor_nid in neighbors:

                    eid = copy_graph.edges_between(nid, neighbor_nid)
                    if isinstance(eid, Iterable):  # multi-graph
                        for e in eid:
                            component_edge_ids.add(e)
                    else:
                        component_edge_ids.add(eid)

                    if neighbor_nid not in visited:
                        visited.add(neighbor_nid)
                        q.append(neighbor_nid)
            # build component
            component_edges = {eid : copy_graph.edges[eid] for eid in component_edge_ids}
            component = self.__class__(component_edges)
            components.append(component)
            to_cover -= set(component.nodes.keys())
        return components

    def shortest_path(self, src, dst, weight):
        """
        Returns the path from src to dst (node ids), using Dijkstra's Algorithm.
        weight is a function that takes in an edge and outputs a weight number.
        """
        V = {nid for nid in self.nodes}
        S = set()
        d = {v:float("inf")
             for v in V
             if v != src}
        d[src] = 0
        prev = {src: None}
        while len(S) < len(V):
            diff_set = V - S
            v = min(diff_set, key=lambda v: d[v])
            S.add(v)
            for eid in self.edges_from(v):
                edge = self.edges[eid]
                w = edge.other(v).id
                cost = weight(edge)
                if d[v] + cost < d[w]:
                    d[w] = d[v] + cost
                    prev[w] = (v, eid)

        # Return a path
        path = []
        if dst not in prev:
            # Path not found
            return None

        pair = prev[dst]
        while pair is not None:
            v, eid = pair
            path.append(eid)
            pair = prev[v]
        return list(reversed(path))


    def subtract(self, other):
        """
        Given another graph "other", produce a graph equivalent
        as subtracting the `other` from this graph. It does not
        matter whether other_map contains nodes that are not in this graph.
        """
        nodes = {}
        edges = {}
        for eid in self.edges:
            if eid not in other.edges:
                edges[eid] = copy.deepcopy(self.edges[eid])
        return self.__class__(edges)


    def partition_by_templates(self, templates,
                               super_node_class=SuperNode, super_edge_class=SuperEdge, **params):
        """
        Invariant: The total number of underlying nodes and edges (i.e. variables) in the result
        is always equal to that in the original graph.
        """
        current_graph = self
        results = {}

        for template in templates:
            print(template.__name__)
            unused_graph, super_graph = current_graph.partition(template, get_unused=True,
                                                                super_node_class=super_node_class,
                                                                super_edge_class=super_edge_class, **params)
            current_graph = unused_graph
            results[template.__name__] = super_graph
        # assert that unused graph is empty and super graph contains the same number of underlying nodes and edges
        count = sum(results[template.__name__].num_nodes() * template.size()
                    for template in templates)
        assert count == self.num_nodes() or (count == self.num_nodes() + self.num_edges())
        return results, count

    def _partition_setup(self):
        edges_available = set(self.edges.keys())
        edges_used = set({})
        nodes_used = set({})  # nodes already used to create new graph
        nodes_available = set(self.nodes.keys()) - nodes_used
        return edges_available, edges_used, nodes_available, nodes_used

    def partition(self, template, get_unused=False,
                  super_node_class=SuperNode, super_edge_class=SuperEdge, **params):
        """
        Partitions this graph by a given template. This algorithm is a realization of the
        Algorithm 2 in http://kaiyuzheng.me/documents/papers/zheng2018aaai.pdf, more
        generalized than Algorithm 1 in http://kaiyuzheng.me/documents/papers/zheng2017thesis.pdf.

        The given template can be of the following types:
        - Only nodes
        - Nodes and edges
        We choose not to support templates with only edges

        Invariant: One edge or one node can only belong to one template (i.e. super node).


        Args:

          template (Template): a Template class. Note that a template is not necessarily a graph.
                               It is generally an EdgeNodeSet, because a valid template can have,
                               for example, only edges and no nodes.
          get_unused (bool): If 'get_unused' is true, return a tuple where the first element
                             is the 'supergraph' and the second element is a graph where nodes
                             are not covered by this partition attempt.
          super_node_class (SuperNode): a SuperNode class that represents
                             the subgraph formed by multiple nodes in the original graph.
          super_edge_class (SuperEdge): a SuperEdge class is an edge added when building the
                             supergraph; it contains functionality to compute the data field
                             for this edge based on the two super nodes on its ends.

        Note that the supergraph will always not contain data on edges, since these data
        may have been covered by some template, which is consolidated into a super node.

        Returns:
          a new Graph that is a 'supergraph' based on this one, or a tuple ('supergraph', 'unused_graph')

        """
        random.seed()
        edges_available, edges_used, nodes_available, nodes_used = self._partition_setup()

        spnodes = {}  # sp: super
        spedges = {}
        spconns = {}

        node_sn = {}  # map from nid to snid
        edge_sn = {}  # map from eid to snid

        while len(edges_available) > 0:
            # randomly sample an edge, then a node on that edge, as the starting point of template matching.
            eid = random.sample(edges_available, 1)[0]
            vindx = random.sample([0,1], 1)[0]
            if len(self.edges[eid].nodes) == 1:
                import pdb; pdb.set_trace()
            nid = self.edges[eid].nodes[vindx].id
            enset = None
            if nid in nodes_used:
                vindx = 1 - vindx
                nid = self.edges[eid].nodes[vindx].id
                if nid in nodes_used:
                    if template.num_nodes() > 0:
                        # This edge is not useful to match this template because both nodes are not present
                        edges_available.remove(eid)
                        continue
                    else:
                        # The given template has no nodes. So it's an edge template. Pass in the sampled edge.
                        enset = template.match(self, None, self.edges[eid],
                                               nodes_used, edges_used, **params)
                        if enset is None:
                            edges_available.remove(eid)
                            continue  # try another edge. It's no use to keep matching on this one.
            if enset is None:
                # The given template can be tried at the sampled node (nid). Try to match it.
                enset = template.match(self, self.nodes[nid], self.edges[eid],
                                       nodes_used, edges_used, **params)
            if enset is None:
                # Unable to match.
                edges_available.remove(eid)
            else:
                # Matched!
                edges_available -= set(enset.edges.keys())
                edges_used |= set(enset.edges.keys())
                nodes_available -= set(enset.nodes.keys())
                nodes_used |= set(enset.nodes.keys())

                # super node
                snid = super_node_class.pick_id(enset, spnodes)
                if snid not in spnodes:
                    spnodes[snid] = super_node_class(snid, enset)
                    spconns[snid] = {}

                # connectivity of super nodes
                for nid in enset.nodes:
                    # check neighbors. If neighbor belongs to a super node, connect the two.
                    for nnid in self.neighbors(nid):
                        if nnid in node_sn:
                            # proof: a proof that the two super nodes are connected. ALWAYS an edge id.
                            #        this is used to deal with multigraphs and mixture of template types.
                            #        See figure for more details.
                            proof = random.sample(self.edges_between(nid, nnid), 1)[0]
                            seid = super_edge_class.pick_id(spnodes[snid], spnodes[node_sn[nnid]],
                                                            existing_ids=set(spedges.keys()))
                            spedges[seid] = super_edge_class(seid, spnodes[snid], spnodes[node_sn[nnid]])
                            if node_sn[nnid] not in spconns[snid]:
                                spconns[snid][node_sn[nnid]] = set({})
                            spconns[snid][node_sn[nnid]].add(proof)
                    node_sn[nid] = snid

                for eid in enset.edges:
                    # check edges outgoing from both ends. If an edge belong to a super node, connect the two.
                    for node in enset.edges[eid].nodes:
                        for eid in self.edges_from(node.id):
                            if eid in edge_sn and edge_sn[eid] in spconns[snid]:
                                # If this edge is not already used as a proof
                                if eid not in spconns[snid][edge_sn[eid]]:
                                    proof = eid
                                    seid = super_edge_class.pick_id(spnodes[snid], spnodes[edge_sn[eid]],
                                                                    existing_ids=set(spedges.keys()))
                                    spedges[seid] = super_edge_class(seid, spnodes[snid], spnodes[edge_sn[eid]])
                                    if edge_sn[eid] not in spconns[snid]:
                                        spconns[snid][edge_sn[eid]] = set({})
                                    spconns[snid][edge_sn[eid]].add(proof)
                    edge_sn[eid] = snid

        # Build the "edges" set for the super graph. It includes spedges, and also degenerate edges for
        # super nodes without connectivity.
        for snid in spnodes:
            if len(spconns[snid]) == 0:
                # No connectivity. Degenerate edge
                seid = super_edge_class.pick_id(spnodes[snid], None,
                                existing_ids=set(spedges.keys()))
                spedges[seid] = super_edge_class(seid, spnodes[snid], None)

        supergraph = Graph(spedges, directed=self._directed)

        if get_unused:
            unused_nodes = {nid: self.nodes[nid] for nid in self.nodes
                            if nid not in nodes_used}
            unused_edges = {eid: self.edges[eid] for eid in self.edges
                            if eid not in edges_used}
            unused_graph = EdgeNodeSet(unused_nodes, unused_edges).to_unused_graph(directed=self._directed,
                                                                                   unused_node_ids=set(unused_nodes.keys()),
                                                                                   unused_edge_ids=set(unused_edges.keys()))
            return unused_graph, supergraph
        else:
            return supergraph


    #--- Conversion ---#
    def to_nx_graph(self):
        import networkx as nx
        if self._directed:
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()

        for eid in self.edges:
            edge = self.edges[eid]
            node1, node2 = edge.nodes
            G.add_edge(node1, node2, **edge.attrs)
        return G

    #--- Visualizations ---#
    def visualize(self, ax, included_nodes=None, dotsize=10, linewidth=1.0,
                  img=None, show_nids=False,
                  **params):  # params = canonical_map_yaml_path=None, consider_placeholders=False,
        """Visualize the topological map `self`. Nodes are colored by labels, if possible.
        If `consider_placeholders` is True, then all placeholders will be colored grey.
        Note that the graph itself may or may not contain placeholders and `consider_placholders`
        is not used to specify that."""

        # Plot the nodes
        for nid in self.nodes:
            if included_nodes is not None and nid not in included_nodes:
                continue

            nid_text = str(nid) if show_nids else None

            node = self.nodes[nid]
            if hasattr(node, "color"):
                node_color = node.color
            else:
                node_color = "grey"
            x, y = node.coords  # gmapping coordinates
            plot_x, plot_y = plot_dot(ax, x, y,
                                      dotsize=dotsize, color=node_color, zorder=2,
                                      linewidth=linewidth, edgecolor='black', label_text=nid_text, alpha=0.6)

            # Plot the edges
            for neighbor_id in self._conns[nid]:
                if included_nodes is not None and neighbor_id not in included_nodes:
                    continue

                edges = self.edges_between(nid, neighbor_id)
                for i, eid in enumerate(edges):
                    edge = self._edges[eid]
                    if hasattr(edge, "color"):
                        edge_color = edge.color
                    else:
                        edge_color = "black"

                    plot_line(ax, node.coords, self.nodes[neighbor_id].coords,
                              linewidth=3, color=edge_color, zorder=1, alpha=0.2)




class UnusedGraph(Graph):
    """
    This is specifically used to deal with "leftover" graphs in after partition.
    This graph may contain nodes and edges that are present for structure but cannot
    actually be used for template matching any more.
    """

    def __init__(self, edges, unused_node_ids=set({}), unused_edge_ids=set({}), directed=False):
        super().__init__(edges, directed=directed)
        self._uu_nids = unused_node_ids
        self._uu_eids = unused_edge_ids

    def _partition_setup(self):
        edges_available = self._uu_eids
        edges_used = set(self.edges.keys()) - set(self._uu_eids)
        nodes_available = self._uu_nids
        nodes_used = set(self.nodes.keys()) - set(self._uu_nids)
        return edges_available, edges_used, nodes_available, nodes_used

    def neighbors(self, node_id):
        """
        Returns a set of neighbor node ids
        """
        return set({nid
                    for nid in set(self._conns[node_id].keys())
                    if nid in self._uu_nids})

    def num_nodes(self):
        return len(self._uu_nids)

    def num_edges(self):
        return len(self._uu_eids)


########################################
# Template
########################################
class Template(ABC):

    @classmethod
    @abstractmethod
    def size(cls):
        """
        A template has a defined size. (e.g. number of nodes/edges).
        Useful for sorting template by complexity.
        """
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        """
        An integer that identifies the type of this template
        """
        pass

    @classmethod
    def templates_for(cls, symbol):
        if symbol.upper() == "THREE":
            return [ThreeNodeTemplate, PairTemplate, SingletonTemplate]
        elif symbol.upper() == "VIEW":
            return [ThreeRelTemplate, SingleRelTemplate, SingleTemplate, RelTemplate]
        elif symbol.upper() == "STAR":
            return [StarTemplate, ThreeNodeTemplate, PairTemplate, SingletonTemplate]
        else:
            raise Exception("Unrecognized symbol for templates: %s" % symbol)

    @classmethod
    def get_type(cls, template):
        if template == ThreeNodeTemplate:
            return "three"
        elif template == StarTemplate:
            return "star"
        elif template == ThreeRelTemplate:
            return "view"
        else:
            raise Exception("Unrecoginzed template to get type: %s" % template.__name__)

    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        """
        pivot_edge (Edge): the edge that was sampled from which the match search starts.
        pivot (Node): the node that was sampled from the pivot_edge from which the match search starts.
        """
        pass

########################################
# NodeTemplate
########################################
class NodeTemplate(Template):

    @classmethod
    @abstractmethod
    def num_nodes(cls):
        pass


    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        pass


    @classmethod
    def size(cls):
        return cls.num_nodes()

    @classmethod
    def code(cls):
        return 0

    has_edge_info = False

    @classmethod
    def size_to_class(num_nodes):
        m = {
            SingletonTemplate.num_nodes: SingletonTemplate,
            PairTemplate.num_nodes: PairTemplate,
            ThreeNodeTemplate.num_nodes: ThreeNodeTemplate,
            StarTemplate.num_nodes: StarTemplate
        }
        return m[num_nodes]


class SingletonTemplate(NodeTemplate):
    """
    Single node
    """
    @classmethod
    def num_nodes(cls):
        return 1

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        # Hardcode the match check
        if pivot.id not in excluded_nodes:
            return OrderedEdgeNodeSet([pivot], [])

class PairTemplate(NodeTemplate):
    """
    Simple pair
    """
    @classmethod
    def num_nodes(cls):
        return 2

    @classmethod
    def match(cls, graph, P, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        """
        Returns an ordered edge node set that follows A-P or P-A.
        """
        # Hardcode the match check
        pivot_neighbors = graph.neighbors(P.id)

        for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
            if A_id not in excluded_nodes and A_id != P.id:
                edge = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                return OrderedEdgeNodeSet([P, graph.nodes[A_id]], [edge])
        return None


class StarTemplate(NodeTemplate):

    """
          A
          |
      B - P - C
          |
          D
    """
    @classmethod
    def num_nodes(cls):
        return 5

    @classmethod
    def match(cls, graph, X, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        def match_by_pivot(P, excluded_nodes=set({})):
            pivot_neighbors = graph.neighbors(P.id)

            #A
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    #B
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            #C
                            for C_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                                if C_id not in excluded_nodes | set({A_id, B_id}):
                                    #D
                                    for D_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                                        if D_id not in excluded_nodes | set({A_id, B_id, C_id}):
                                            edge_PA = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                                            edge_PB = graph.edges[random.sample(graph.edges_between(P.id, B_id), 1)[0]]
                                            edge_PC = graph.edges[random.sample(graph.edges_between(P.id, C_id), 1)[0]]
                                            edge_PD = graph.edges[random.sample(graph.edges_between(P.id, D_id), 1)[0]]
                                            return OrderedEdgeNodeSet([graph.nodes[A_id], graph.nodes[B_id], P, graph.nodes[C_id], graph.nodes[D_id]],
                                                                        [edge_PA, edge_PB, edge_PC, edge_PD])

        subgraph = match_by_pivot(X, excluded_nodes=excluded_nodes)
        return subgraph    # subgraph could be None



class ThreeNodeTemplate(NodeTemplate):

    """
    Simple three node structure

    A--(P)--B

    P is the pivot. A and B are not connected
    """

    @classmethod
    def num_nodes(cls):
        return 3

    @classmethod
    def match(cls, graph, X, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        """
        Returns a list of node ids that are matched in this template. Order
        follows A-P-B, where P is the pivot node. X is the node where the
        matching starts, but not necessarily the pivot.
        """
        def match_by_pivot(P, excluded_nodes=set({})):
            pivot_neighbors = graph.neighbors(P.id)
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            # if not relax and A_id in graph.neighbors(B_id):
                            #     continue
                            edge_PA = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                            edge_PB = graph.edges[random.sample(graph.edges_between(P.id, B_id), 1)[0]]
                            return OrderedEdgeNodeSet([graph.nodes[A_id], P, graph.nodes[B_id]],
                                                      [edge_PA, edge_PB])


        subgraph = match_by_pivot(X, excluded_nodes=excluded_nodes)
        return subgraph


class EdgeRelTemplate(Template):

    @classmethod
    @abstractmethod
    def num_edges(cls):
        pass

    @classmethod
    @abstractmethod
    def num_nodes(cls):
        pass

    @classmethod
    @abstractmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        pass


    @classmethod
    def size(cls):
        return cls.num_edges() + cls.num_nodes()

    @classmethod
    def code(cls):
        return 1

    has_edge_info = False

    @classmethod
    def size_to_class(num_nodes, num_edges):
        m = {
            # SingletonTemplate.num_nodes: SingletonTemplate,
            # PairTemplate.num_nodes: PairTemplate,
            # ThreeNodeTemplate.num_nodes: ThreeNodeTemplate,
            # StarTemplate.num_nodes: StarTemplate
        }
        return m[num_nodes]



class ThreeRelTemplate(EdgeRelTemplate):

    @classmethod
    def num_edges(cls):
        return 2

    @classmethod
    def num_nodes(cls):
        return 3

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):

        def match_by_pivot(P, excluded_nodes=set({}), excluded_edges=set({})):
            pivot_neighbors = graph.neighbors(P.id)
            for A_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                if A_id not in excluded_nodes:
                    for B_id in random.sample(pivot_neighbors, len(pivot_neighbors)):
                        if B_id not in excluded_nodes | set({A_id}):
                            # if not relax and A_id in graph.neighbors(B_id):
                            #     continue
                            edge_PA = graph.edges[random.sample(graph.edges_between(P.id, A_id), 1)[0]]
                            edge_PB = graph.edges[random.sample(graph.edges_between(P.id, B_id), 1)[0]]

                            if edge_PA.id not in excluded_edges and edge_PB.id not in excluded_edges:
                                return OrderedEdgeNodeSet([graph.nodes[A_id], P, graph.nodes[B_id]],
                                                            [edge_PA, edge_PB])
        return match_by_pivot(pivot, excluded_nodes=excluded_nodes, excluded_edges=excluded_edges)

class SingleRelTemplate(EdgeRelTemplate):

    @classmethod
    def num_edges(cls):
        return 1

    @classmethod
    def num_nodes(cls):
        return 1

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        if pivot is not None and pivot.id not in excluded_nodes\
           and pivot_edge.id not in excluded_edges:
            return OrderedEdgeNodeSet([pivot], [pivot_edge])

class RelTemplate(EdgeRelTemplate):
    @classmethod
    def num_edges(cls):
        return 1

    @classmethod
    def num_nodes(cls):
        return 0

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        if pivot_edge.id not in excluded_edges:
            return OrderedEdgeNodeSet([], [pivot_edge])

class SingleTemplate(EdgeRelTemplate):
    @classmethod
    def num_edges(cls):
        return 0

    @classmethod
    def num_nodes(cls):
        return 1

    @classmethod
    def match(cls, graph, pivot, pivot_edge, excluded_nodes=set({}), excluded_edges=set({}), **params):
        if pivot is not None and pivot.id not in excluded_nodes:
            return OrderedEdgeNodeSet([pivot], [])


#########################
# Build Graph From File #
#########################
def build_graph(graph_file_path, func_interpret_node, func_interpret_edge):
    """
    Reads a file that is a specification of a graph, then construct
    a graph object using the TopologicalMap class. The TopologicalMap
    class is for undirected graphs, where each node on the graph contains
    a fixed label and edges do not have labels. It is possible for
    nodes to have uncertain labels.

    This script reads a file of format ".ug" that indicates "undirected graph".
    The format is:


    <Node_Id> <attr1> <attr2> ...
    --
    <Edge_Id> <Node_Id_1> <Node_Id_2> <attr1> <attr2> ...
    --
    Undirected

    The first part specifies the nodes. It is not required that Node_Id
    starts from 0. Each node may have a list of attributes, which are
    interpreted using the given `func_interpret_node` function, that takes
    as input node id, [<attr1>, <attr2> ...], and returns an
    object of (sub)class of Node.

    The second part specifies the edges. The edge is undirected, and
    the node ids should be defined in the first part of the file.
    Each edge may have a list of attributes, which are
    interpreted using the given `func_interpret_edge` function, that takes
    as input the edge id, node1, node2 (objects) and [<attr1>, <attr2> ...], and
    returns an object of (sub)class of Node. For simplicity, you can omit <Edge_Id>
    by ":" which causes edge_id to be incremental from 0.

    Lastly, one can specify whether the graph is "directed" or "undirected".
    If unspecified, assume to be "undirected".

    There could be arbitrarily many empty lines, and can have comments
    by beginning the line with "#"

    This function can be used to parse a graph file and generate a Graph
    object, be used in GraphSPN experiments.
    """
    with open(graph_file_path) as f:
        lines = f.readlines()

    nodes = {}  # Map from node id to an actual node object
    edges = {}  # Map from edge id to an actual edge object
    use_log = None

    directed = False

    state = "nodes"
    for i, line in enumerate(lines):
        # Handle transition, if encountered
        try:
            line = line.rstrip()
            if len(line) == 0:
                continue # blank line
            if line.startswith("#"):
                continue # comment

            if line == "--":
                state = _next_state(state)
                continue # read next line
            # This line belongs to a state
            elif state == "nodes":
                tokens = line.split()  # split on whitespaces
                nid = int(tokens[0])  # split on whitespaces
                if nid in nodes:
                    raise ValueError("Node %d is already defined" % (nid))
                nodes[nid] = func_interpret_node(nid, tokens[1:])
            elif state == "edges":
                tokens = line.split()  # split on whitespaces
                if tokens[0] == ":":
                    eid = len(edges)
                else:
                    eid = int(tokens[0])
                nid1, nid2 = int(tokens[1]), int(tokens[2])
                if eid in edges:
                    raise ValueError("Edge %d is already defined" % (eid))
                if nid1 not in nodes:
                    raise ValueError("Node %d is undefined" % nid1)
                if nid2 not in nodes:
                    raise ValueError("Node %d is undefined" % nid2)

                edges[eid] = func_interpret_edge(eid, nodes[nid1], nodes[nid2], tokens[3:])

            elif state == "graph_type":
                directed = False if line.startswith("Undirected") else True
            else:
                raise ValueError("Unexpected state %s" % state)
        except Exception as e:
            print("Line %d caused an Error:" % i)
            print(e)
            raise e

    return Graph(edges, directed=directed) # We are done

#####################
# Utility functions #
#####################
def _next_state(state):
    if state == "nodes":
        return "edges"
    elif state == "edges":
        return "graph_type"
    else:
        raise ValueError("Unexpected state %s" % state)


##########Topo-map related##########
def compute_view_number(node, neighbor, divisions=8):
    """
    Assume node and neighbor have the 'pose' attribute. Return an integer
    within [0, divisions-1] indicating the view number of node that is
    connected to neighbor.
    """
    x, y = node.coords[0], node.coords[1]
    nx, ny = neighbor.coords[0], neighbor.coords[1]
    angle_rad = math.atan2(ny-y, nx-x)
    if angle_rad < 0:
        angle_rad = math.pi*2 - abs(angle_rad)
    view_number = int(math.floor(angle_rad / (2*math.pi / divisions)))  # floor division
    return view_number


__all__ = [
    'Node',
    'Edge',
    'SuperNode',
    'SuperEdge',
    'EdgeNodeSet',
    'OrderedEdgeNodeSet',
    'UnusedGraph',
    'Graph',
    'Template',
    'NodeTemplate',
    'SingletonTemplate',
    'StarTemplate',
    'ThreeNodeTemplate',
    'EdgeRelTemplate',
    'ThreeRelTemplate',
    'SingleRelTemplate',
    'RelTemplate',
    'SingleTemplate',
    'build_graph',
    'compute_view_number'
]


################# Plotting ##################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
import matplotlib.patheffects as path_effects


def transform_coordinates(gx, gy, map_spec, img):
    # Given point (gx, gy) in the gmapping coordinate system (in meters), convert it
    # to a point or pixel in Cairo context. Cairo coordinates origin is at top-left, while
    # gmapping has coordinates origin at lower-left.
    imgHeight, imgWidth = img.shape
    res = float(map_spec['resolution'])
    originX = float(map_spec['origin'][0])  # gmapping map origin
    originY = float(map_spec['origin'][1])
    # Transform from gmapping coordinates to pixel cooridnates.
    return ((gx - originX) / res, imgHeight - (gy - originY) / res)

def plot_dot(ax, px, py, color='blue', dotsize=2, fill=True, zorder=0, linewidth=0, edgecolor=None, label_text=None, alpha=1.0):
    very_center = plt.Circle((px, py), dotsize, facecolor=color, fill=fill, zorder=zorder, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha)
    ax.add_artist(very_center)
    if label_text:
        text = ax.text(px, py, label_text, color='white',
                        ha='center', va='center', size=7, weight='bold')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])

        # t = ax.text(px-5, py-5, label_text, fontdict=font)
        # t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))
    return px, py

def plot_line(ax, p1, p2, linewidth=1, color='black', zorder=0, alpha=1.0):
    p1x, p1y = p1
    p2x, p2y = p2
    ax = plt.gca()
    line = lines.Line2D([p1x, p2x], [p1y, p2y], linewidth=linewidth, color=color, zorder=zorder,
                        alpha=alpha)
    ax.add_line(line)


def zoom_plot(p, img, ax, zoom_level=0.35):
    # Zoom by setting limits. Center around p
    px, py = p
    h, w = img.shape
    sidelen = min(w*zoom_level*0.2, h*zoom_level*0.2)
    ax.set_xlim(px - sidelen/2, px + sidelen/2)
    ax.set_ylim(py - sidelen/2, py + sidelen/2)


def zoom_rect(p, img, ax, h_zoom_level=0.35, v_zoom_level=0.35):
    # Zoom by setting limits
    px, py = p
    h, w = img.shape
    xsidelen = w*h_zoom_level*0.2
    ysidelen = h*v_zoom_level*0.2
    ax.set_xlim(px - xsidelen/2, px + xsidelen/2)
    ax.set_ylim(py - ysidelen/2, py + ysidelen/2)


def plot_to_file(*args, labels=[], path="plot.png", xlabel=None, ylabel=None):
    """
    Plot data in *args to a file specified by path. If
    path is None, just save to plot.png locally.
    """
    for i, data in enumerate(args):
        if i < len(labels):
            plt.plot(data, label=labels[i])
        else:
            plt.plot(data)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close()


def plot_roc(roc_data, savepath='roc.png', names=[]):
    """roc_data is list of tuples (fpr, tpr)"""
    import sklearn.metrics
    from pylab import rcParams
    rcParams['figure.figsize'] = 4, 4
    for i, item in enumerate(roc_data):
        fpr, tpr = item
        name = names[i] if len(names) > i else "Model%d" % i
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' %
                 (name, sklearn.metrics.auc(fpr, tpr)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
