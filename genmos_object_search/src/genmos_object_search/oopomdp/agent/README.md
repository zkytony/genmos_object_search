Since the pursuit of genmos_object_search as a gRPC server,
the way an agent should be constructed is refactored.

However, I don't want to mess up previously working code.
Previously working code are now under oopomdp/deprecated.

Note that the models in oopomdp/models are still directly
applicable in building the agents.

Here, the base class of all agents is MosAgent. Variants
such as MosAgentBasic2D, MosAgentBasic3D, etc. are child
classes of MosAgent.  The class SloopMosAgent is a child
class of SloopAgent, which is a SLOOP (Spatial Language Object
Oriented POMDP) applied specifically for multi-object
search (in 2D or 3D).

"GenMOS agent" refers to a variant of MosAgent without
SLOOP, by default, MosAgentTopo3D. A hierarchical GenMOS
agent is a MosAgentTopo2D that can instantiate a
MosAgentTopo3D when it decides to search locally. This
instantiation logic is not implemented here; it is
implemented in the GenMOS server, since that requires
additional input (sensor data) from the client.
