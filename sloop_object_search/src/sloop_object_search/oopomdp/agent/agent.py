import pomdp_py
from ..models.search_region import SearchRegion2D
from . import belief

class Mos2DAgent(pomdp_py.Agent):
    """The top-level class for 2D agent. A 2D agent is
    one who believes objects lie on a 2D plane, and it
    carries a 2D sensor (e.g. fan-shaped sensor).

    The action space and transition model are not specified here."""
    def __init__(self, agent_config, search_region,
                 init_robot_belief,
                 init_object_beliefs=None):
        """
        Args:
            agent_config (dict): configuration for the agent
            search_region (SearchRegion2D): 2d search region
            init_robot_belief (pomdp_py.GenerativeDistribution): belief over robot state
            init_object_beliefs (dict): maps from object id to pomdp_py.GenerativeDistribution
        """
        assert isinstance(search_region, SearchRegion2D),\
            "search region of a 2D agent should of type SearchRegion2D."
        self.agent_config = agent_config
        self.search_region = search_region
        robot = agent_config["robot"]
        objects = agent_config["objects"]
        self.robot_id = robot['id']
        self.target_objects = {target_id: objects[target_id]
                               for target_id in self.agent_config["targets"]}

        # Belief
        if init_object_beliefs is None:
            init_object_beliefs = belief.init_object_beliefs(
                self.target_objects, self.search_region,
                prior=self.agent_config["belief"].get("prior", {}))
        init_belief = pomdp_py.OOBelief({robot_id: init_robot_belief,
                                         **init_object_beliefs})
