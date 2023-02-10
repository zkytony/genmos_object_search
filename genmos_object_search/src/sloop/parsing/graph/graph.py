"""Base classes of Node, Edge, and Graph.

**NOTE**:
The classes Node, Edge, EdgeNodeSet and Graph are adapted
from graphspn: github.com/zkytony/graphspn.

The nodes here can have coordinates, used for plotting.
"""

class Node:
    """A node, which could have an (x,y) location"""
    def __init__(self, id, data=None):
        """
        The id is expected to be unique in the graph.
        """
        self.id = id
        self.data = data

    def __repr__(self):
        return "%s(%d)" % (type(self).__name__, self.id)

    def __hash__(self):
        return hash(self.id)


class Edge:
    """An edge links two nodes."""
    def __init__(self, id, node1, node2, data=None):
        """
        The id is expected to be unique in the graph edges.
        """
        self.id = id
        if node2 is None:
            self.nodes = (node1,)
        else:
            self.nodes = (node1, node2)

        self.data = data

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

    @property
    def color(self):
        return "black"


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
                    raise ValueError("Edge %s is degenerate" % str(edge))
                if endpoints[1].id not in nodes:
                    nodes[endpoints[1].id] = endpoints[1]
            edges = {e.id:e for e in edges}
        elif type(edges) == dict:
            for eid in edges:
                endpoints = edges[eid].nodes
                if endpoints[0].id not in nodes:
                    nodes[endpoints[0].id] = endpoints[0]
                if edges[eid].degenerate:  # degenerate edge; only one node.
                    raise ValueError("Edge %s is degenerate" % str(edge))
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
    
    def _build_connections(self):
        """Builds the self._conns field, which is a map from node id to a dictionary of neighbor id -> edge(s).
        Implementation differs between different types of graphs."""
        self._multi = False
        for eid in self._edges:
            if self._edges[eid].degenerate:  # degenerate, only one node.
                raise ValueError("Edge %s is degenerate" % str(edge))
            else:
                node1, node2 = self._edges[eid].nodes
            if node1.id not in self._conns:
                self._conns[node1.id] = {}
                self._outedges[node1.id] = set({})
            if self._edges[eid].degenerate:  # degenerate, only one node.
                raise ValueError("Edge %s is degenerate" % str(edge))
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
