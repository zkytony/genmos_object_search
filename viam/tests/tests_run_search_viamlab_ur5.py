# Code written specifically for the test at Viam Lab on the UR5 robot
#
# Viam Robot Pose
# Connected!
# (0.2797589770640316, 0.7128048233719448, 0.5942370817926967, -0.6500191634979094, 0.4769735333791088, 0.4926158987104014, 0.32756817897816304)
# position {
#   x: 0.27975897706403158
#   y: 0.71280482337194484
#   z: 0.59423708179269674
# }
# rotation {
#   x: -0.65001916349790945
#   y: 0.4769735333791088
#   z: 0.4926158987104014
#   w: 0.32756817897816304
# }
#
# Example output of object detection
# header {
# }
# robot_id: "robot0"
# detections {
#   label: "Chair"
#   box_2d {
#     x_min: 69
#     y_min: 146
#     x_max: 161
#     y_max: 237
#   }
# }
# detections {
#   label: "Person"
#   box_2d {
#     x_min: 14
#     y_min: 87
#     x_max: 38
#     y_max: 127
#   }
# }
# detections {
#   label: "Person"
#   box_2d {
#     x_min: 195
#     y_min: 31
#     x_max: 226
#     y_max: 104
#   }
# }



########### main object search logic ###########
async def run_sloop_search(viam_robot,
                           config,
                           world_frame=None,
                           dynamic_update=False):
    """config: configuration dictionary for SLOOP"""
    sloop_client = SloopObjectSearchClient()
    agent_config = config["agent_config"]
    planner_config = config["planner_config"]
    robot_id = agent_config["robot"]["id"]

    last_action = None
    objects_found = set()
    #-----------------------------------------

    # First, create an agent
    sloop_client.createAgent(
        header=proto_utils.make_header(), config=agent_config,
        robot_id=robot_id)

    # Make the client listen to server
    ls_future = sloop_client.listenToServer(
        robot_id, server_message_callback)
    local_robot_id = None  # needed if the planner is hierarchical

    # Update search region
    update_search_region(robot_id, agent_config, sloop_client)

    # wait for agent creation
    print("waiting for sloop agent creation...")
    sloop_client.waitForAgentCreation(robot_id)
    print("agent created!")

    # # visualize initial belief
    # get_and_visualize_belief()

    # # create planner
    # response = sloop_client.createPlanner(config=planner_config,
    #                                       header=proto_utils.make_header(),
    #                                       robot_id=robot_id)
    # rospy.loginfo("planner created!")

    # # Send planning requests
    # for step in range(config["task_config"]["max_steps"]):
    #     action_id, action_pb = plan_action()
    #     execute_action(action_id, action_pb)

    #     if dynamic_update:
    #         update_search_region(robot_id, agent_config, sloop_client)

    #     response_observation, response_robot_belief =\
    #         self.wait_observation_and_update_belief(action_id)
    #     print(f"Step {step} robot belief:")
    #     robot_belief_pb = response_robot_belief.robot_belief
    #     objects_found = set(robot_belief_pb.objects_found.object_ids)
    #     objects_found.update(objects_found)
    #     print(f"  pose: {robot_belief_pb.pose.pose_3d}")
    #     print(f"  objects found: {objects_found}")
    #     print("-----------")

    #     # visualize FOV and belief
    #     self.get_and_visualize_belief()
    #     if response_observation.HasField("fovs"):
    #         self.visualize_fovs_3d(response_observation)

    #     # Check if we are done
    #     if objects_found == set(self.agent_config["targets"]):
    #         rospy.loginfo("Done!")
    #         break
    #     time.sleep(1)



async def test_ur5e_viamlab():
    with open("./config/ur5_exp1_viamlab.yaml") as f:
        config = yaml.safe_load(f)

    print(">>>>>>><<<<<<<<>>>> viam connecting >><<<<<<<<>>>>>>>")
    ur5robot = await connect_viamlab_ur5()
    print('Resources:')
    print(ur5robot.resource_names)

    print(">>>>>>><<<<<<<<>>>> begin >><<<<<<<<>>>>>>>")
    await run_sloop_search(ur5robot,
                           config,
                           world_frame="arm_origin",
                           dynamic_update=False)



if __name__ == "__main__":
    asyncio.run(test_ur5e_viamlab())
