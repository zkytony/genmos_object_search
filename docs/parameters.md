Below is the table of all parameters that is considered in the GenMOS system
that is general (i.e. not middleware-specific).

Each role of the table corresponds to an entry in the configuration dictionary.
The string in "Category" corresponds to the (sequence of) keys leading up
to the actual parameter, and the string in "Name of Parameter" corresponds to,
of course, the parameter name. For example, if "Category" is `a.b` and
"Name of Parameter" is `c`, then this corresponds to a dictionary structure
as follows:
```
"a": {
   "b": {
      "c": <parameter_value>
   }
}
```



To understand the information in this table:
* In `Category`, a string `a.b`


| **Category**               | **Name of Parameter** | **Description**                                                                                                                                                    | **Example**                                                      |
|----------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| agent_config               | agent_class           | Python class name of the GenMOS POMDP agent.                                                                                                                       | MosAgentBasic2D, MosAgentBasic3D, MosAgentTopo2D, MosAgentTopo3D |
| agent_config               | agent_type            | Type of agent (in the planning hierarchy). Intended for differentiating an agent that focuses on local search and an agent that can deal with hierarchical search. | Must be 'local', 'hierarchical', or 'local_hierarchical'         |
| agent_config.search_region |                       |                                                                                                                                                                    |                                                                  |
