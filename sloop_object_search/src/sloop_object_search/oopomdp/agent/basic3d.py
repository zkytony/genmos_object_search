"""This represents a local 3D search agent with a primitive action
space such as moving / rotating with respect to an axis.
"""
import logging
import pomdp_py
import numpy as np
from ..domain.observation\
    import JointObservation, ObjectDetection, Voxel, FovVoxels
from ..models import belief
from ..models.transition_model import RobotTransBasic3D
from ..models.sensors import FrustumCamera
from ..models.policy_model import PolicyModelBasic3D
from ..models.octree_belief import update_octree_belief
from .common import MosAgent,\
    init_object_transition_models, init_primitive_movements, init_detection_models
from genmos_object_search.utils import math as math_utils

class MosAgentBasic3D(MosAgent):

    def init_detection_models(self):
        # Check that all sensors have the same look direction.
        # Otherwise, the robot needs a primary camera direction.
        detection_models = init_detection_models(self.agent_config)
        self.default_forward_direction = None
        for d in detection_models:
            camera_model = detection_models[d].sensor
            assert isinstance(camera_model, FrustumCamera),\
                "For now, 3D agent sensor should be FrustumCamera"
            if self.default_forward_direction is None:
                self.default_forward_direction = camera_model.look
            else:
                if self.default_forward_direction != camera_model.look:
                    try:
                        self.default_forward_direction\
                            = self.agent_config["robot"]["default_forward_direction"]
                    except KeyError:
                        raise ValueError("robot has multiple cameras that look in different directions."
                                         "Requires specifying 'default_primary_camera_direction' to know"
                                         "the robot's default forward direction")
        return detection_models

    def init_transition_and_policy_models(self):
        robot_trans_model = RobotTransBasic3D(
            self.robot_id, self.reachable,
            self.detection_models,
            no_look=self.no_look,
            default_forward_direction=self.default_forward_direction)
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # TODO: allow for sampling local search viewpoints
        target_ids = self.agent_config["targets"]
        action_config = self.agent_config["robot"]["action"]
        primitive_movements = init_primitive_movements(action_config)
        policy_model = PolicyModelBasic3D(target_ids,
                                          robot_trans_model,
                                          primitive_movements)
        return transition_model, policy_model

    def reachable(self, pos):
        """A position is reachable if it is a valid
        voxel and it is not occupied. Assume 'pos' is a
        position at the ground resolution level"""
        above_ground = pos[2] >= self.reachable_config.get("min_height", 0)
        not_too_high = pos[2] <= self.reachable_config.get("max_height", float('inf'))
        valid_voxel = self.search_region.octree_dist.octree.valid_voxel(*pos, 1)
        not_occupied = not self.search_region.occupied_at(pos, res=1)
        return above_ground and not_too_high and valid_voxel and not_occupied

    def _update_object_beliefs(self, observation, action=None, debug=False, **kwargs):
        assert isinstance(observation, JointObservation)
        if not self.robot_id in observation:
            raise ValueError("requires knowing robot pose corresponding"\
                             " to the object detections.")

        robot_pose = observation.z(self.robot_id).pose
        return_fov = kwargs.get("return_fov", False)
        return_vobz = kwargs.get("return_volumetric_observations", False)

        fovs = {}  # maps from objid to (visible_volume, obstacles_hit)
        vobzs = {}
        for objid in observation:
            if objid == self.robot_id:
                continue
            objo = observation.z(objid)
            if not isinstance(objo, ObjectDetection):
                raise NotImplementedError(f"Unable to handle object observation of type {type(objo)}")

            # construct a volumetric observation about this object.
            detection_model = self.detection_models[objid]
            params = self.agent_config["belief"].get("visible_volume_params", {})
            fov_voxels, visible_volume, obstacles_hit =\
                build_volumetric_observation(objo, detection_model.sensor,
                    robot_pose, self.search_region.octree_dist, **params)

            # Now, we finally update belief, if objid is a target object, each
            # voxel is used to update belief.
            if objid in self.target_objects:
                b_obj = self.belief.b(objid)
                if debug:
                    _visualize_octree_belief(b_obj, robot_pose, occupancy_octree=self.search_region.octree_dist,
                                             visible_volume=visible_volume, obstacles_hit=obstacles_hit)
                alpha = detection_model.alpha
                beta = detection_model.beta
                b_obj_new = update_octree_belief(b_obj, fov_voxels, alpha, beta)
                if debug:
                    _visualize_octree_belief(b_obj_new, robot_pose)
                self.belief.set_object_belief(objid, b_obj_new)
            else:
                # objid is not a target object. It may be a correlated object each
                # voxel will inform the belief update of surrounding voxels.
                raise NotImplementedError("Doesn't handle correlated object right now.")

            # save fov used for belief update
            fovs[objid] = (visible_volume, obstacles_hit)
            vobzs[objid] = fov_voxels

        aux = {}
        if return_fov:
            aux["fovs"] = fovs
        if return_vobz:
            aux["vobzs"] = vobzs
        return aux


def build_volumetric_observation(detection, camera_model, robot_pose, occupancy_octree,
                                 **params):
    """Return a FovVoxels object as the representation of volumetric observation,
    built based on an object detection, the camera model, robot_pose, obstacles
    modeled by occupancy octree.

    Also returns (visible_volume, obstacles_hit).
    """
    assert isinstance(detection, ObjectDetection)
    visible_volume, obstacles_hit = camera_model.visible_volume(
        robot_pose, occupancy_octree,
        return_obstacles_hit=True, **params)

    # Note: if the voxels are bigger, this shouldn't be that slow.
    # we will label voxels that
    voxels = {}  # maps from voxel to label
    overlapped = False
    detected = False
    for voxel in visible_volume:
        # voxel should by (x,y,z,r)
        if detection.pose == ObjectDetection.NULL:
            voxels[voxel] = Voxel(voxel, Voxel.FREE)
        else:
            detected = True
            if detection.pose == ObjectDetection.NO_POSE:
                # this is a label-only detection. All voxels within FOV will be marked.
                voxels[voxel] = Voxel(voxel, detection.id)
                overlapped = True
            else:
                # this is a 3d bbox detection. Check if the detection bbox overlaps with voxel
                x,y,z,r = voxel
                voxel_box = ((x*r, y*r, z*r), r, r, r)
                bbox = detection.bbox_axis_aligned()
                if math_utils.boxes_overlap3d_origin(bbox, voxel_box):
                    voxels[voxel] = Voxel(voxel, detection.id)
                    overlapped = True
                else:
                    voxels[voxel] = Voxel(voxel, Voxel.FREE)

    if detected and not overlapped:
        print(f"Warning: detected object {detection.id} at {detection.loc} "\
              "but it is not in agent's FOV model.")

    return FovVoxels(voxels), visible_volume, obstacles_hit


#### useful debugging methods
try:
    import open3d as o3d
    from genmos_object_search.utils.open3d_utils\
        import draw_octree_dist, cube_unfilled, draw_robot_pose, draw_fov
    from genmos_object_search.utils.colors import cmaps
except OSError as ex:
    logging.error("Failed to load open3d: {}".format(ex))

def _visualize_octree_belief(octree_belief, robot_pose, occupancy_octree=None,
                             visible_volume=None, obstacles_hit=None):
    geometries = []
    if occupancy_octree is not None:
        geometries = draw_octree_dist(occupancy_octree, viz=False)
    # Draw the robot
    geometries.append(draw_robot_pose(robot_pose))
    # Draw the octree belief
    geometries.extend(draw_octree_dist(octree_belief.octree_dist, viz=False,
                                       cmap=cmaps.COLOR_MAP_GRAYS))
    # Draw the FOV
    if visible_volume is not None:
        geometries.extend(draw_fov(visible_volume, obstacles_hit))
    o3d.visualization.draw_geometries(geometries)
