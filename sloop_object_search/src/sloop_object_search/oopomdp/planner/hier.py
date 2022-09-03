import pomdp_py
from ..agent.topo2d import MosAgentTopo2D

class HierarchicalPlanner(pomdp_py.Planner):
    """
    Hierarchical planner. Interacts with a global search agent,
    and manages a local search agent, if necessary. The global
    search agent is a 2D topo agent (topo2d.py), while the local search agent
    is a 3D topo agent (topo3d.py).
    """
    def __init__(self, topo_agent, **planner_params):
        if not isinstance(topo_agent, MosAgentTopo2D):
            raise TypeError("HierarchicalPlanner requires global search "\
                            "agent to be of type MosAgentTopo2D, but got "\
                            f"{type(topo_agent)}")
        print("HELLO")
        print(planner_params)
