import os
import torch
import pomdp_py
from .osm.models.heuristics.rules import BASIC_RULES
from .osm.models.heuristics.model import MixtureSLUModel
from .osm.datasets import MapInfoDataset

class SloopAgent(pomdp_py.Agent):
    def __init__(self,
                 agent_config,
                 map_name):
        self.robot_id = agent_config["robot"]["id"]
        self.map_name = map_name
        self.mapinfo = MapInfoDataset()
        self.map_info.load_by_name(self.map_name)
        # Spatial Language Observation Model
        self.spacy_model = spacy.load(agent_config.get(
            "spacy_model", "en_core_web_lg"))
        self.splang_observation_model =\
            self._init_splang_observation_model(
                agent_config["foref_models_dir"],
                device=agent_config.get("device", "cuda:0"))
        self._init_oopomdp()

    def _init_oopomdp(self):
        """Should call the super().__init__ method
        to actually create the pomdp_py.Agent"""
        raise NotImplementedError

    def _init_splang_observation_model(self,
                                       foref_models_dir,
                                       device="cpu"):
        def _mp(predicate):
            """model path"""
            return os.path.join(foref_models_dir,
                                "iter%s_%s:%s:%s"
                                % (iteration, "ego-ctx-foref-angle",
                                   predicate, self.map_name),
                                "%s_model.pt" % predicate)

        device = torch.device(device if torch.cuda.is_available() else "cpu")
        foref_models = {
            "front": torch.load(_mp("front"), map_location=device),
            "behind": torch.load(_mp("front"), map_location=device),
            "left": torch.load(_mp("left"), map_location=device),
            "right": torch.load(_mp("right"), map_location=device),
        }
        return MixtureSLUModel(BASIC_RULES,
                               self.mapinfo,
                               foref_models=foref_models,
                               foref_kwargs={"device": device})

    def update(self, observation, action):
        next_robot_state = robot_state_from_obz(observation.z(self.robot_id))
        new_object_beliefs = {}
        for objid in self.object_beliefs:
            if objid == self.robot_id:
                self.belief.set_object_belief(
                    objid, pomdp_py.Histogram({next_robot_state: 1.0}))
            else:
                assert isinstance(agent.transition_model.transition_models[objid],
                                  StaticObjectTransitionModel)

                obseravtion_model = agent.observation_model
                if isinstance(observation, SpatialLanguageObservation):
                    self.splang_observation_model.set_object_id(objid)
                    obseravtion_model = self.splang_observation_model

                self.belief.update_object_belief(
                    objid, observation.z(objid), action, agent, obseravtion_model)
