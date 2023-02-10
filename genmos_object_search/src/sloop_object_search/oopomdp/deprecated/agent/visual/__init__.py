def visualize_step(viz, agent, env, action, _config, **kwargs):
    objlocs = {j: env.state.s(j).loc
               for j in env.state.object_states
               if j != agent.robot_id}
    colors = {j: _config["agent_config"]["objects"][j].get("color", [128, 128, 128])
              for j in env.state.object_states
              if j != agent.robot_id}
    no_look = _config["agent_config"]["no_look"]

    draw_fov = list(objlocs.keys())
    if not no_look:
        if not isinstance(action, LookAction):
            draw_fov = None
    if action is None:
        draw_fov = None
    viz.visualize(agent, objlocs, colors=colors, draw_fov=draw_fov, **kwargs)
