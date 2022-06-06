import os
import torch
import pomdp_py
import spacy
from .osm.models.heuristics.rules import BASIC_RULES
from .osm.models.heuristics.model import MixtureSLUModel
from .osm.datasets import MapInfoDataset
from .observation import SpatialLanguageObservation

class SloopAgent(pomdp_py.Agent):
    def __init__(self,
                 agent_config,
                 map_name):
        """
        agent_config (dict): specifies various configurations.
            See example in tests/src/sloop_object_search/test_sloop_system.py
        """
        self.agent_config = agent_config
        self.robot_id = agent_config["robot"]["id"]
        self.map_name = map_name
        self.mapinfo = MapInfoDataset()
        self.mapinfo.load_by_name(self.map_name)
        # Spatial Language Observation Model
        self.spacy_model = spacy.load(agent_config.get(
            "spacy_model", "en_core_web_lg"))
        self.splang_observation_model =\
            self._init_splang_observation_model(
                agent_config["foref_models_dir"],
                device=agent_config.get("device", "cuda:0"),
                symbol_map=agent_config.get("object_symbol_map", None))

        pomdp_components = self._init_oopomdp()
        super().__init__(*pomdp_components)


    def _init_oopomdp(self):
        """Should call the super().__init__ method
        to actually create the pomdp_py.Agent"""
        raise NotImplementedError

    def _init_splang_observation_model(self,
                                       foref_models_dir,
                                       device="cpu",
                                       symbol_map=None):
        def _mp(predicate):
            """model path"""
            iteration = 2
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
            "right": torch.load(_mp("left"), map_location=device),
        }
        return MixtureSLUModel(BASIC_RULES,
                               self.mapinfo,
                               foref_models=foref_models,
                               foref_kwargs={"device": device},
                               symbol_map=symbol_map)

    def update_belief(self, observation, action):
        self.belief.update_robot_belief(observation, action)
        next_robot_state = self.belief.mpe().s(self.robot_id)
        for objid in self.belief.object_beliefs:
            if objid == self.robot_id:
                continue
            else:
                obseravtion_model = self.observation_model
                if isinstance(observation, SpatialLanguageObservation):
                    self.splang_observation_model.set_object_id(objid)
                    obseravtion_model = self.splang_observation_model

                self.belief.update_object_belief(self,
                    objid, observation,
                    next_robot_state, action,
                    obseravtion_model)
