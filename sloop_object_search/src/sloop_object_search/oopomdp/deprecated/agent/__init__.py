from .basic2d import SloopMosBasic2DAgent, MosBasic2DAgent
from .basic3d import MosBasic3DAgent
from .visual.basic2d import VizSloopMosBasic2D
from .topo2d import SloopMosTopo2DAgent
from .visual.topo2d import VizSloopMosTopo
from .agent import make_agent

# agent whose model of the world is 2D
AGENT_CLASS_2D = {
    "SloopMosBasic2DAgent",
    "SloopMosTopo2DAgent",
    "MosBasic2DAgent",
}

# agent whose model of the world is 3D
AGENT_CLASS_3D = {
    "MosBasic3DAgent",
}
