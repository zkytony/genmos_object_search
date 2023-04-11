Below is the table of all parameters that is considered in the GenMOS system
that is general (i.e. not middleware-specific).

| **Category** | **Name of Parameter** | **Description**                                                                                                                                                    | **Example**                                                      |
|--------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| agent_config | agent_class           | Python class name of the GenMOS POMDP agent.                                                                                                                       | MosAgentBasic2D, MosAgentBasic3D, MosAgentTopo2D, MosAgentTopo3D |
|              | agent_type            | Type of agent (in the planning hierarchy). Intended for differentiating an agent that focuses on local search and an agent that can deal with hierarchical search. | Must be 'local', 'hierarchical', or 'local_hierarchical'         |
|              |                       |                                                                                                                                                                    |                                                                  |
