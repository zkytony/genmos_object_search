import pomdp_py
from sloop_object_search.utils.misc import import_class
from sloop_object_search.oopomdp.agent import HierMosAgent
from sloop_object_search.oopomdp.domain.action import StayAction
from sloop_object_search.oopomdp.domain.observation import RobotLocalization
from ..constants import Message, Info

def create_planner(planner_config, agent):
    """
    Creates a planner specified by 'planner_config' for
    the 'agent' (pomdp_py.Agent).
    """
    planner_class = planner_config["planner"]
    if planner_class.endswith("POUCT")\
       or planner_class.endswith("POMCP"):
        planner = import_class(planner_class)(**planner_config["planner_params"],
                                              rollout_policy=agent.policy_model)
        return planner
    elif planner_class.endswith("HierPlanner"):
        planner = import_class(planner_class)(agent, **planner_config["planner_params"])
        return planner
    else:
        raise NotImplementedError(f"Planner {planner_class} is unrecognized.")


def plan_action(planner, agent, server):
    """
    Performs planning and outputs the next action for execution.

    Note for hierarchical agent:
    When the planner of the global search agent outputs StayAction,
    the the following happens:
      (1) the server sends the client a message requesting a UpdateSearchRegion
      (2) the server waits, until receiving this message from the client.
      (3) Upon receiving this message, a search_region is created, and
          a *local search agent* is also then created. This should be a 3D agent.
          Note that the client does not need to supply configuration for
          the creation of this client. This configuration is interpreted based
          on the initial configuration when creating the ~HierMosAgent~. The
          initial belief of the local search agent is based on the global
          search agent's belief.
      (4) A planner for the local search agent is created.
      (5) Planning is performed for the local search agent. The action is
          returned for execution to the client. This means the client
          doesn't see the "StayAction" by the global agent; It will take
          an action corresponding to the local search agent's planning output.

    Args:
        planner (pomdp_py.Planner)
        agent (MosAgent)
        server (grpc server)
    Returns:
        (bool, str or Action)
        If planning is successful, return True, and the action.
        If failed, return False, and a string as the reason
    """
    action = planner.plan(agent)
    if hasattr(agent, "tree") and agent.tree is not None:
        # print planning tree
        _dd = pomdp_py.utils.TreeDebugger(agent.tree)
        _dd.p(0)

    # If the agent is hierarchical, and it planned a Stay
    # action, then will ask the server to wait for an
    # update search region request in order to gather
    # the necessary inputs to create the local agent.
    if isinstance(agent, HierMosAgent)\
       and isinstance(action, StayAction):
        # server tells client, please send over update search region request
        server.add_message(agent.robot_id,
                           Message.REQUEST_SEARCH_REGION_UPDATE.format(agent.robot_id))
        local_search_region, robot_loc_world =\
            server.wait_for_client_provided_info(
                Info.LOCAL_SEARCH_REGION_INFO.format(agent.robot_id))
        robot_pose_local, robot_pose_cov_local =\
            local_search_region.to_pomdp_pose(robot_loc_world.pose, robot_loc_world.cov)
        robot_loc_local = RobotLocalization(
            agent.robot_id, robot_pose_local,
            robot_pose_cov_local)
        agent.create_local_agent(local_search_region, robot_loc_local)

        print("Local agent created")

    return True, action
