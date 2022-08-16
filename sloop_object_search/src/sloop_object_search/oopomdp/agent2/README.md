Since the pursuit of sloop_object_search as a gRPC server,
the way an agent should be constructed is refactored. However,
I don't want to mess up previously working code. Therefore,
I am adding 'agent2' that contain the class interface, similar
in spirit to GridMap2.

Note that the models in oopomdp/models are still directly
applicable in building the agents.
