import pomdp_py
from . import belief
from ..domain.observation import JointObservation, ObjectDetection, Voxel, FovVoxels
from ..models.transition_model import RobotTransBasic3D
from ..models.policy_model import PolicyModelBasic3D
from ..models.octree_belief import update_octree_belief
from .common import MosAgent, SloopMosAgent,\
    init_object_transition_models, init_primitive_movements
from sloop_object_search.utils import math as math_utils

class MosAgentBasic3D(MosAgent):

    def __init__(self, agent_config, search_region,
                 init_robot_pose_dist,
                 init_object_beliefs=None):
        super().__init__(agent_config, search_region,
                         init_robot_pose_dist,
                         init_object_beliefs=init_object_beliefs)

    def init_transition_and_policy_models(self):
        robot_trans_model = RobotTransBasic3D(
            self.robot_id, self.reachable,
            self.detection_models,
            no_look=self.no_look)
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
        return self.search_region.octree.valid_voxel(*pos, 1)\
            and not self.search_region.occupied_at(pos, res=1)

    def update_belief(self, observation, action=None):
        """
        update belief given observation. If the observation is
        object detections, we expect it to be of type JointObservation,
        """
        if isinstance(observation, JointObservation):
            if not self.robot_id in observation:
                raise ValueError("requires knowing robot pose corresponding"\
                                 " to the object detections.")

            robot_pose = observation.z(self.robot_id).pose
            visible_volume = None  # we will
            for objid in observation:
                if objid == self.robot_id:
                    continue
                objo = observation.z(objid)
                if not isinstance(objo, ObjectDetection):
                    raise NotImplementedError(f"Unable to handle object observation of type {type(objo)}")

                # We will construct a volumetric observation about this
                # object. If the observation is about a target object, each
                # voxel is used to update belief. If the observation is about a
                # correlated object, then each voxel will inform the belief
                # update of surrounding voxels.
                detection_model = self.detection_models[objid]
                params = self.agent_config["belief"].get("visible_volume_params", {})
                visible_volume = detection_model.sensor.visible_volume(
                    robot_pose, self.search_region.octree_dist, **params)
                # Note: if the voxels are bigger, this shouldn't be that slow.
                # we will label voxels that
                voxels = {}  # maps from voxel to label
                overlapped = True
                for voxel in visible_volume:
                    # voxel should by (x,y,z,r)
                    if objo.pose == ObjectDetection.NULL:
                        voxels[voxel] = Voxel(voxel, Voxel.FREE)
                    else:
                        x,y,z,r = voxel
                        bbox = objo.bbox_axis_aligned
                        voxel_box = ((x*r, y*r, z*r), r, r, r)
                        if math_utils.boxes_overlap3d_origin(bbox, voxel_box):
                            voxels[voxel] = objid
                            overlapped = True
                        else:
                            voxels[voxel] = Voxel(voxel, Voxel.FREE)

                if not overlapped:
                    print(f"Warning: detected object {objid} but it is not in agent's FOV model.")

                # Now, we finally update belief, if objid is a target object
                if objid in self.target_objects:
                    b_obj = self.belief.b(objid)
                    b_obj_new = update_octree_belief(b_obj, FovVoxels(voxels),
                                                     alpha=detection_model.alpha,
                                                     beta=detection_model.beta)
                    self.belief.set_object_belief(objid, b_obj_new)

                else:
                    # objid is not a target object. It may be a correlated object
                    raise NotImplementedError("Doesn't handle correlated object right now.")
