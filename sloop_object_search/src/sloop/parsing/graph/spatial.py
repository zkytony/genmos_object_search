import json
import matplotlib.pyplot as plt
from .graph import Node, Edge, Graph
from . import util

# # For visualization
# import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout

class SpatialEntity(Node):
    def __init__(self, id, name):
        self.id = id
        self.data = name

    @property
    def name(self):
        return self.data

    def __repr__(self):
        return "#%d(%s)" % (self.id, self.name)


class SpatialRelation(Edge):
    def __init__(self, id, node1, node2, label):
        if not isinstance(node1, SpatialEntity)\
           or not isinstance(node2, SpatialEntity):
            raise ValueError("node1 and node2 must both be SpatialEntity")
        super().__init__(id, node1, node2, data=label)

    @property
    def label(self):
        return self.data

    @label.setter
    def label(self, value):
        self.data = value

    def __repr__(self):
        return "#%d[<%s>%s<%s>]" % (self.id,
                                    str(self.nodes[0]),
                                    str(self.label),
                                    str(self.nodes[1]))


class SpatialGraph(Graph):
    """
    A Spatial Graph is a graph where nodes are
    spatial entities and edges indicate spatial relations.

    Ideally, a spatial graph can be constructed from
    a natural language, and be used to construct
    egocentric representations.
    """
    def __init__(self, lang, edges, lang_original=None):
        self.lang = lang  # The possibly symbolized language
        self.lang_original = lang_original  # the original, unsymbolized language
        super().__init__(edges)

    @classmethod
    def from_dict(cls, dct):
        """Constructs spatial graph from dictionary"""
        nodes = {}
        for name in dct["entities"]:
            nodes[name] = SpatialEntity(len(nodes), name)
        edges = set({})
        for name1, name2, label in dct["relations"]:
            node1, node2 = nodes[name1], nodes[name2]
            edges.add(SpatialRelation(len(edges), node1, node2, label))
        if "lang_original" in dct:
            return SpatialGraph(dct["lang"], edges, dct["lang_original"])
        else:
            return SpatialGraph(dct["lang"], edges)

    def to_dict(self):
        dct = {"entities": [], "relations": [], "lang": self.lang}
        for nid in sorted(self.nodes):
            dct["entities"].append(self.nodes[nid].name)
        for eid  in sorted(self.edges):
            edge = self.edges[eid]
            node1, node2 = edge.nodes
            dct["relations"].append((node1.name, node2.name, edge.label))
        return dct

    def to_file(self, fpath, **kwargs):
        """Writes this spatial graph to a json file."""
        with open(fpath, 'w') as f:
            dct = self.to_dict()
            dct.update(dict(kwargs))
            json.dump(dct, f, indent=4, sort_keys=True)

    def __repr__(self):
        return str(self.edges)

    @classmethod
    def join(cls, sg_list):
        """Joins mulitple spatial graphs together into a single one.
        We assume that the entities with the same name refers to actually
        the same object or landmark - therefore they refer to the same
        spatial entity node."""
        dct = {"entities": set({}), "relations": set({}), "lang": ""}
        for i, sg in enumerate(sg_list):
            dct["lang"] += sg.lang
            if i < len(sg_list) - 1:
                dct["lang"] += ". "
            for eid in sg.edges:
                node1, node2 = sg.edges[eid].nodes
                dct["entities"].add(node1.name)
                dct["entities"].add(node2.name)
                dct["relations"].add((node1.name, node2.name, sg.edges[eid].label))
        return SpatialGraph.from_dict(dct)
