import numpy as np

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.any_pb2 import Any
import logging
import pickle

import pomdp_py

import sloop_object_search.grpc.observation_pb2 as o_pb2
from sloop_object_search.grpc.common_pb2\
    import Vec2, Vec3, Header, Pose2D, Pose3D, Quaternion, Histogram, Voxel3D
from sloop_object_search.grpc.action_pb2\
    import MoveViewpoint, Find, KeyValueAction, Motion2D, Motion3D
from .. import sloop_object_search_pb2 as slpb2

from sloop_object_search.oopomdp.domain import action as slpa
from sloop_object_search.oopomdp.domain import observation as slpo
from sloop_object_search.oopomdp.models.belief import RobotStateBelief
from sloop_object_search.oopomdp.models.search_region import SearchRegion3D
from sloop_object_search.oopomdp.models.octree_belief\
    import Octree, OctreeBelief, plot_octree_belief
from sloop_object_search.utils import math as math_utils
from sloop_object_search.utils import open3d_utils

def v3toa(v3):
    """convert Vec3 proto to numpy array"""
    return np.array([v3.x, v3.y, v3.z])

def process_search_region_params_2d(search_region_params_2d_pb):
    params = {}
    if search_region_params_2d_pb.HasField('layout_cut'):
        params["layout_cut"] = search_region_params_2d_pb.layout_cut
    if search_region_params_2d_pb.HasField('floor_cut'):
        params["floor_cut"] = search_region_params_2d_pb.floor_cut
    if search_region_params_2d_pb.HasField('grid_size'):
        params["grid_size"] = search_region_params_2d_pb.grid_size
    if search_region_params_2d_pb.HasField('brush_size'):
        params["brush_size"] = search_region_params_2d_pb.brush_size
    if search_region_params_2d_pb.HasField('region_size'):
        params["region_size"] = search_region_params_2d_pb.region_size
    if search_region_params_2d_pb.HasField('debug'):
        params["debug"] = search_region_params_2d_pb.debug
    return params

def process_search_region_params_3d(search_region_params_3d_pb):
    params = {}
    if search_region_params_3d_pb.HasField('octree_size'):
        params["octree_size"] = search_region_params_3d_pb.octree_size
    if search_region_params_3d_pb.HasField('search_space_resolution'):
        params["search_space_resolution"] = search_region_params_3d_pb.search_space_resolution
    if search_region_params_3d_pb.HasField('debug'):
        params["debug"] = search_region_params_3d_pb.debug
    if search_region_params_3d_pb.HasField('region_size_x'):
        params["region_size_x"] = search_region_params_3d_pb.region_size_x
    if search_region_params_3d_pb.HasField('region_size_y'):
        params["region_size_y"] = search_region_params_3d_pb.region_size_y
    if search_region_params_3d_pb.HasField('region_size_z'):
        params["region_size_z"] = search_region_params_3d_pb.region_size_z
    return params


def pointcloudproto_to_array(point_cloud):
    points_array = np.array([[p.pos.x, p.pos.y, p.pos.z]
                             for p in point_cloud.points])
    return points_array

def posemsg_to_pose3dproto(pose_msg):
    return Pose3D(position=Vec3(x=pose_msg.position.x,
                                y=pose_msg.position.y,
                                z=pose_msg.position.z),
                  rotation=Quaternion(x=pose_msg.orientation.x,
                                      y=pose_msg.orientation.y,
                                      z=pose_msg.orientation.z,
                                      w=pose_msg.orientation.w))

def posetuple_to_poseproto(pose):
    if len(pose) == 7:
        x,y,z,qx,qy,qz,qw = pose
        return Pose3D(position=Vec3(x=x, y=y, z=z),
                      rotation=Quaternion(x=qx, y=qy, z=qz, w=qw))
    elif len(pose) == 3:
        x,y,th = pose
        return Pose2D(x=x, y=y, th=th)
    else:
        raise ValueError(f"Invalid pose: {pose}")

def poseproto_to_posetuple(pose_pb):
    if isinstance(pose_pb, Pose2D):
        return (pose_pb.x, pose_pb.y, pose_pb.th)
    elif isinstance(pose_pb, Pose3D):
        return (pose_pb.position.x,
                pose_pb.position.y,
                pose_pb.position.z,
                pose_pb.rotation.x,
                pose_pb.rotation.y,
                pose_pb.rotation.z,
                pose_pb.rotation.w)
    else:
        raise ValueError(f"Invalid pose proto: {pose_pb}")

def quatproto_to_tuple(quat):
    return (quat.x, quat.y, quat.z, quat.w)

def make_header(frame_id=None, stamp=None):
    if stamp is None:
        stamp = Timestamp().GetCurrentTime()
    if frame_id is None:
        return Header(stamp=stamp)
    else:
        return Header(stamp=stamp, frame_id=frame_id)

def robot_pose_from_proto(robot_pose_pb, include_cov=False):
    """returns a tuple representation from RobotPose proto.

    If it is 2D, then return (x, y, th). If it is 3D, then return (x, y, z, qx,
    qy, qz, qw).

    Note that this doesn't change the frame.

    If include_cov is True, then return a covariance
    matrix in the robot_pose_pb.
    """
    if not isinstance(robot_pose_pb, o_pb2.RobotPose):
        raise TypeError("robot_pose_pb should be RobotPose")
    if robot_pose_pb.HasField("pose_2d"):
        pose2d = robot_pose_pb.pose_2d
        pose = (pose2d.x, pose2d.y, pose2d.th)

        if include_cov:
            if len(robot_pose_pb.covariance) > 0:
                assert len(robot_pose_pb.covariance) == 9,\
                    "covariance matrix for 2D pose in proto should be a 9-element double list"
                cov = np.array(robot_pose_pb.covariance).reshape(3, 3)
            else:
                cov = np.zeros((3, 3))

    elif robot_pose_pb.HasField("pose_3d"):
        pose3d = robot_pose_pb.pose_3d
        pose = (pose3d.position.x, pose3d.position.y, pose3d.position.z,
                pose3d.rotation.x, pose3d.rotation.y, pose3d.rotation.z,
                pose3d.rotation.w)

        if include_cov:
            if len(robot_pose_pb.covariance) > 0:
                assert len(robot_pose_pb.covariance) == 36,\
                    "covariance matrix for 3D pose in proto should be a 36-element double list"
                cov = np.array(robot_pose_pb.covariance).reshape(6, 6)
            else:
                cov = np.zeros((6, 6))

    else:
        raise ValueError("request does not contain valid robot pose field.")

    if include_cov:
        return pose, cov
    else:
        return pose

def robot_localization_from_proto(robot_pose_pb):
    """Returns a RobotLocalization object from RobotPose proto.
    Note that this is just a type conversion; it doesn't assume
    which frame robot_pose_pb is in"""
    pose, cov = robot_pose_from_proto(robot_pose_pb, include_cov=True)
    return slpo.RobotLocalization(robot_pose_pb.robot_id, pose, cov)

def robot_pose_proto_from_tuple(robot_pose):
    """Returns a RobotPose proto from a given tuple
    representation of robot pose. Note that robot_pose
    is assumed to contain both position and rotation."""
    if len(robot_pose) == 3:
        return o_pb2.RobotPose(pose_2d=posetuple_to_poseproto(robot_pose))
    elif len(robot_pose) == 7:
        return o_pb2.RobotPose(pose_3d=posetuple_to_poseproto(robot_pose))
    else:
        raise ValueError(f"Invalid pose: {robot_pose}")

def pomdp_action_to_proto(action, agent, header=None, **kwargs):
    if header is None:
        header = make_header()
    if isinstance(action, slpa.MotionAction):
        action_type = "move_action"
        if isinstance(action, slpa.MotionAction2D):
            forward, angle = action.motion
            motion_pb = Motion2D(
                forward=forward * agent.search_region.search_space_resolution,
                dth=angle)
            action_pb = MoveViewpoint(header=header,
                                      robot_id=agent.robot_id,
                                      motion_2d=motion_pb,
                                      name=action.name,
                                      expected_cost=action.step_cost)

        elif isinstance(action, slpa.MotionActionTopo):
            # then sample N random states from agent's belief, then
            # use the resulting pose as the goal viewpoint -> this
            # results in a rotation that faces the target while respecting
            # the agent's current belief.
            robot_trans_model = agent.transition_model[agent.robot_id]
            rnd_state = agent.belief.random()
            goal_pose_pomdp = robot_trans_model.sample(rnd_state, action).pose
            goal_pose_world = agent.search_region.to_world_pose(goal_pose_pomdp)
            goal_pose_world_pb = posetuple_to_poseproto(goal_pose_world)
            if isinstance(goal_pose_world_pb, Pose2D):
                dest = {"dest_2d": goal_pose_world_pb}
            else:
                dest = {"dest_3d": goal_pose_world_pb}
            action_pb = MoveViewpoint(header=header,
                                      robot_id=agent.robot_id,
                                      name=action.name,
                                      expected_cost=action.step_cost,
                                      **dest)

        elif isinstance(action, slpa.MotionAction3D):
            # we need to convert the position change from pomdp frame to
            # the world frame.
            dpos_pomdp, drot = action.motion
            dpos_world = agent.search_region.to_world_dpos(dpos_pomdp)
            motion_pb = Motion3D(
                dpos=Vec3(x=dpos_world[0], y=dpos_world[1], z=dpos_world[2]),
                drot_euler=Vec3(x=drot[0], y=drot[1], z=drot[2]))
            action_pb = MoveViewpoint(header=header,
                                      robot_id=agent.robot_id,
                                      motion_3d=motion_pb,
                                      name=action.name,
                                      expected_cost=action.step_cost)
    elif isinstance(action, slpa.FindAction):
        action_type = "find_action"
        action_pb = Find(header=header,
                         robot_id=agent.robot_id,
                         name=action.name)
    else:
        raise RuntimeError(f"Unrecognized action {action}")
    return action_type, action_pb


def interpret_planned_action(plan_action_reply):
    """Given the response from PlanActionReply,
    return the protobuf object corresponding to
    the action."""
    assert isinstance(plan_action_reply, slpb2.PlanActionReply),\
        "only interprets PlanActionReply"
    if plan_action_reply.HasField("move_action"):
        return plan_action_reply.move_action
    elif plan_action_reply.HasField("find_action"):
        return plan_action_reply.find_action
    elif plan_action_reply.HasField("kv_action"):
        return plan_action_reply.kv_action
    else:
        raise ValueError("unable to determine action.")


def pomdp_object_beliefs_to_proto(object_beliefs, search_region):
    """
    Args:
        object_beliefs; Maps from objid to a pomdp_py.GenerativeDistribution
    Return:
        A list of ObjectBelief protos
    """
    object_beliefs_proto = []
    for objid in object_beliefs:
        b_obj = object_beliefs[objid]

        hist_values = []  # the search region locations
        hist_probs = []   # the probabilities
        if not isinstance(search_region, SearchRegion3D):
            # For 2D belief, just iterate over all
            for loc in b_obj.loc_dist:
                x, y = search_region.to_world_pos(loc)
                hist_values.append(to_any_proto(Vec2(x=x, y=y)))
                hist_probs.append(b_obj.loc_dist.prob_at(*loc))

        else:
            # b_obj is octree belief
            assert isinstance(b_obj, OctreeBelief),\
                "3d object belief should be octree belief"

            # each voxel is (x,y,z,r,_) where x,y,z are ground-level voxel coordinates.
            # open3d_utils.draw_octree_dist(b_obj.octree_dist)
            voxels = b_obj.octree_dist.collect_plotting_voxels()
            probs = [b_obj.octree_dist.prob_at(*Octree.increase_res(voxels[i][:3], 1, voxels[i][3]), voxels[i][3])
                     for i in range(len(voxels))]
            for i in range(len(voxels)):
                vpos = voxels[i][:3]  # voxel location at ground-level (but in pomdp frame)
                vres = voxels[i][3]
                x, y, z = search_region.to_world_pos(vpos)
                res = vres * search_region.search_space_resolution
                voxel_pb = Voxel3D(pos=Vec3(x=x, y=y, z=z), res=res)
                hist_values.append(to_any_proto(voxel_pb))
                hist_probs.append(probs[i])

        dist = Histogram(length=len(hist_values),
                         values=hist_values,
                         probs=hist_probs)
        object_beliefs_proto.append(
            slpb2.ObjectBelief(object_id=objid,
                               dist=dist))
    return object_beliefs_proto


def to_any_proto(val):
    val_any = Any()
    val_any.Pack(val)
    return val_any


def robot_belief_to_proto(robot_belief, search_region, header=None, **other_fields):
    """Given a robot belief (in POMDP frame), return a RobotBelief proto
    (in world frame). Uncertainty over the robot belief is possibly
    in its pose. The robot observes its other attributes such as 'objects_found'."""
    if not isinstance(robot_belief, RobotStateBelief):
        raise TypeError("robot_belief should be a RobotStateBelief")
    if header is None:
        header = make_header()

    mpe_robot_state = robot_belief.mpe()
    robot_id = mpe_robot_state["id"]

    # get robot pose in world frame
    robot_pose_pomdp = robot_belief.pose_est.mean
    robot_pose_cov_pomdp = robot_belief.pose_est.covariance
    robot_pose_world, robot_pose_cov_world =\
        search_region.to_world_pose(robot_pose_pomdp, robot_pose_cov_pomdp)
    if mpe_robot_state.is_2d:
        pose_field = {"pose_2d": posetuple_to_poseproto(robot_pose_world)}
    else:
        pose_field = {"pose_3d": posetuple_to_poseproto(robot_pose_world)}
    cov_flat = np.asarray(robot_pose_cov_world).flatten()
    robot_pose_pb = o_pb2.RobotPose(header=header, robot_id=robot_id,
                                    covariance=cov_flat, **pose_field)
    # objects found
    objects_found_pb = o_pb2.ObjectsFound(
        header=header, robot_id=robot_id,
        object_ids=list(map(str, mpe_robot_state.objects_found)))
    return slpb2.RobotBelief(robot_id=robot_id,
                             objects_found=objects_found_pb,
                             pose=robot_pose_pb,
                             **other_fields)

def pomdp_detection_from_proto(detection_pb, agent,
                               pos_precision='int',
                               rot_precision=0.001,
                               size_precision=0.001):
    """given Detection3D proto, return ObjectDetection object
    The pose in the detection will be converted to POMDP space.
    Its position and orientation will be rounded to the specified
    precision."""
    objid = detection_pb.label
    center = detection_pb.box.center
    sizes = v3toa(detection_pb.box.sizes)

    # because the POMDP frame and the world frame are axis-aligned,
    # we only need to convert the position, not rotation.
    search_region = agent.search_region
    if not agent.is_hierarchical and not search_region.is_3d:
        # agent is local 2D agent
        center_pos = (center.position.x, center.position.y)
        quat = quatproto_to_tuple(center.rotation)
        _, _, yaw = math_utils.quat_to_euler(*quat)
        center_rot = math_utils.fround(rot_precision, yaw)
        pomdp_center_pos = math_utils.fround(pos_precision, search_region.to_pomdp_pos(center_pos))
        pomdp_sizes = math_utils.fround(size_precision, sizes[:2] / search_region.search_space_resolution)
    else:
        # agent is 3D or hierarchical. Will receive 3D observation
        center_pos = (center.position.x, center.position.y, center.position.z)
        center_rot = math_utils.fround(rot_precision, quatproto_to_tuple(center.rotation))
        pomdp_center_pos = math_utils.fround(pos_precision, search_region.to_pomdp_pos(center_pos))
        pomdp_sizes = math_utils.fround(size_precision, sizes / search_region.search_space_resolution)

    pomdp_pose = (pomdp_center_pos, center_rot)
    return slpo.ObjectDetection(objid, pomdp_pose, sizes=pomdp_sizes)

def pomdp_robot_observation_from_request(request, agent, action=None,
                                         pos_precision='int',
                                         rot_precision=0.001):
    """Create a RobotObservation object from request."""
    # This is a bit tricky so single it out. Basically, need to set
    # 'camera_direction' field properly to respect the 'no_look' attribute
    # of the agent (i.e. whether the agent actively decides whether to
    # process observation about objects now).
    assert isinstance(request, slpb2.ProcessObservationRequest),\
        "request must be ProcessObservationRequest"
    if request.robot_id != agent.robot_id:
        raise ValueError("request is not for the agent")

    if not agent.no_look:
        if action is None or not isinstance(action, slpa.LookAction):
            camera_direction = None  # agent has Look action, but didn't take it.
        else:
            camera_direction = action.name
    else:
        # agent accepts receiving detections
        camera_direction = "look"  # reward model expects this to be set in order to reward "Find"
    # Now, create robot localization
    robot_pose_world, robot_pose_world_cov = robot_pose_from_proto(
        request.robot_pose, include_cov=True)
    robot_pose_pomdp, robot_pose_pomdp_cov =\
        agent.search_region.to_pomdp_pose(robot_pose_world, robot_pose_world_cov)
    if len(robot_pose_pomdp) == 7:
        # rounding - eliminate numerical issues for planner update
        robot_pos_pomdp = math_utils.fround(pos_precision, robot_pose_pomdp[:3])
        robot_rot_pomdp = math_utils.fround(rot_precision, robot_pose_pomdp[3:])
    else:
        robot_pos_pomdp = math_utils.fround(pos_precision, robot_pose_pomdp[:2])
        robot_rot_pomdp = math_utils.fround(rot_precision, robot_pose_pomdp[2:])
    robot_pose_pomdp = (*robot_pos_pomdp, *robot_rot_pomdp)

    robot_pose_estimate_pomdp = slpo.RobotLocalization(
        agent.robot_id, robot_pose_pomdp, robot_pose_pomdp_cov)

    if not agent.search_region.is_3d and not agent.is_hierarchical:
        # 2D local agent receives 2D observation
        robot_pose_estimate_pomdp = robot_pose_estimate_pomdp.to_2d()

    # objects found
    objects_found = set(agent.belief.b(agent.robot_id).mpe().objects_found)  # objects already found
    if request.HasField("objects_found"):
        objects_found |= set(request.objects_found.object_ids)
    objects_found = tuple(sorted(objects_found))

    # Now create the robot observation object; Note that even for
    # topo agents, they will only receive RobotObservation instead
    # of RobotObservationTopo because the client has no obligation
    # to maintain the topo map or node ids.
    robot_observation = slpo.RobotObservation(agent.robot_id,
                                              robot_pose_estimate_pomdp,
                                              objects_found,
                                              camera_direction)
    return robot_observation

def pomdp_observation_from_request(request, agent, action=None):
    """Given a ProcessObservationRequest, and the
    agent this request is for, return a pomdp Observation
    observation that represents that observation contained
    in the request."""
    assert isinstance(request, slpb2.ProcessObservationRequest),\
        "request must be ProcessObservationRequest"
    if request.robot_id != agent.robot_id:
        raise ValueError("request is not for the agent")

    # we will always create a robot observation
    robot_observation = pomdp_robot_observation_from_request(request, agent, action=action)

    if request.HasField("object_detections"):
        # Create JointObservation based on object detections
        # We will make an ObjectDetection observation for
        # every detectable object. if the object isn't detected,
        # its pose is NULL.
        # First collect what we do detect
        detections = {}
        for detection_pb in request.object_detections.detections:
            zobj = pomdp_detection_from_proto(detection_pb, agent)
            if zobj.id not in detections:
                detections[zobj.id] = zobj
            else:
                logging.warning(f"multiple detections for {zobj.id}. Only keeping one.")

        # Now go through every detectable object
        zobjs = {agent.robot_id: robot_observation}
        for objid in agent.detection_models:
            if objid not in detections:
                zobj = slpo.ObjectDetection(objid, slpo.ObjectDetection.NULL)
            else:
                zobj = detections[objid]
            zobjs[objid] = zobj
        return slpo.JointObservation(zobjs)

    elif request.HasField("language"):
        raise NotImplementedError("Not there yet")

    return robot_observation

def topo_map_to_proto(topo_map, search_region):
    # list of TopoEdge protos
    edges_pb = []
    for eid in topo_map.edges:
        edge = topo_map.edges[eid]
        if not edge.degenerate:
            node1, node2 = edge.nodes
            node1_pb = topo_node_to_proto(node1, search_region)
            node2_pb = topo_node_to_proto(node2, search_region)
            edge_pb = slpb2.TopoEdge(id=str(eid),
                                     node1=node1_pb,
                                     node2=node2_pb)
            edges_pb.append(edge_pb)
    return slpb2.TopoMap(edges=edges_pb)


def topo_node_to_proto(node, search_region):
    pos_world = search_region.to_world_pos(node.pos)
    if len(pos_world) == 2:
        pos = {"pos_2d": Vec2(x=pos_world[0], y=pos_world[1])}
    else:
        pos = {"pos_3d": Vec3(x=pos_world[0], y=pos_world[1], z=pos_world[2])}
    node_pb = slpb2.TopoNode(id=str(node.id), **pos)
    return node_pb

def pos_from_topo_node(node_pb):
    """given TopoNode, return a tuple for its pos"""
    if node_pb.HasField("pos_3d"):
        return (node_pb.pos_3d.x,
                node_pb.pos_3d.y,
                node_pb.pos_3d.z)
    else:
        return (node_pb.pos_2d.x,
                node_pb.pos_2d.y)
