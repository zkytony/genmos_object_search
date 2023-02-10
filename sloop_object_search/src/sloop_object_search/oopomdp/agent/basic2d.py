# basic2d: 2D search agent with primitive action space.
import logging
import pomdp_py
from tqdm import tqdm
from ..models.search_region import SearchRegion2D
from ..models.transition_model import RobotTransBasic2D
from ..models.policy_model import PolicyModelBasic2D
from ..models import belief
from ..domain.observation import JointObservation, ObjectDetection, FovVoxels, Voxel
from .common import MosAgent, init_object_transition_models, init_primitive_movements
# open3d requires Ubuntu 18.04+ with GLIBC 2.27+
try:
    from genmos_object_search.utils import open3d_utils
except OSError as ex:
    logging.error("Failed to load open3d: {}".format(ex))


class MosAgentBasic2D(MosAgent):

    def init_transition_and_policy_models(self):
        # transition model
        trans_args = self.agent_config["robot"].get("transition", {})
        robot_trans_model = RobotTransBasic2D(
            self.robot_id, self.detection_models,
            self.reachable, self.no_look, **trans_args
        )
        object_transition_models = {
            self.robot_id: robot_trans_model,
            **init_object_transition_models(self.agent_config)}
        transition_model = pomdp_py.OOTransitionModel(object_transition_models)

        # policy model
        target_ids = self.agent_config["targets"]
        action_config = self.agent_config["robot"]["action"]
        primitive_movements = init_primitive_movements(action_config)
        policy_model = PolicyModelBasic2D(
            target_ids, robot_trans_model, primitive_movements)
        return transition_model, policy_model

    def reachable(self, pos):
        return pos not in self.search_region.grid_map.obstacles

    @property
    def is_3d(self):
        return False

    def _update_object_beliefs(self, observation, action=None, debug=False, **kwargs):
        assert isinstance(observation, JointObservation)
        if not self.robot_id in observation:
            raise ValueError("requires knowing robot pose corresponding"\
                             " to the object detections.")

        robot_pose_est = observation.z(self.robot_id).pose_est
        robot_pose = robot_pose_est.pose
        if robot_pose_est.is_3d:
            raise ValueError("2D agent should receive 2D robot observation")

        for objid in observation:
            if objid == self.robot_id:
                continue
            zobj = observation.z(objid)
            if not isinstance(zobj, ObjectDetection):
                raise NotImplementedError(f"Unable to handle object observation of type {type(zobj)}")
            if zobj.loc is not None and zobj.is_3d:
                raise ValueError("2D agent should receive 2D object observation")
            # build area observation
            detection_model = self.detection_models[objid]
            label_only = detection_model.label_only\
                if hasattr(detection_model, "label_only") else False
            fov_cells = build_area_observation(
                zobj, detection_model.sensor, robot_pose, label_only=label_only)
            # TODO: potentially set these parameters dynamically
            if objid in self.target_objects:
                b_obj = self.belief.b(objid)
                alpha = detection_model.alpha
                beta = detection_model.beta
                b_obj_new = belief.update_object_belief_2d(
                    b_obj, fov_cells, alpha, beta)
                self.belief.set_object_belief(objid, b_obj_new)
            else:
                # objid is not a target object. It may be a correlated object each
                # voxel will inform the belief update of surrounding voxels.
                raise NotImplementedError("Doesn't handle correlated object right now.")


def build_area_observation(detection, fansensor, robot_pose,
                           num_samples=1000, label_only=False,
                           debug=False, discrete=True):
    """Returns a set of ((x,y), label) tuples where (x,y) is
    a location in the FOV. label=detection.id if (x,y) is
    the detection's location, or if 'label_only' is True.
    label='free' otherwise."""
    cells = {}
    for i in tqdm(range(num_samples), desc="Building area FOV"):
        x, y = fansensor.uniform_sample_sensor_region(robot_pose)
        grid_x, grid_y = int(round(x)), int(round(y))
        if not fansensor.in_range((grid_x, grid_y), robot_pose):
            continue
        label = 'free'
        if label_only:
            if detection.loc is not None:
                label = detection.id
        else:
            if (grid_x, grid_y) == detection.loc:
                label = detection.id

        if discrete:
            cell_loc = (grid_x, grid_y)
        else:
            cell_loc = (x, y)
        if cell_loc in cells:
            if cells[cell_loc] == 'free' and label != 'free':
                cells[cell_loc] = label
        else:
            cells[cell_loc] = label
    cells = {(k, v) for k, v in cells.items()}
    if debug:
        open3d_utils.draw_fov_2d(cells, robot_pose, viz=True)
    return cells

def project_fov_voxels_to_2d(fov_voxels, search_region3d, search_region2d, objid=None):
    if not isinstance(fov_voxels, FovVoxels):
        raise TypeError(f"fov_voxels: Expect FovVoxels. Got {type(fov_voxels)}")
    # Project voxels down to 2D
    fov_cells = {}
    for voxel_pos in fov_voxels.voxels:
        voxel = fov_voxels.voxels[voxel_pos]
        if len(voxel_pos) == 3:
            voxel_pos = (*voxel_pos, 1)
        res = voxel_pos[-1]
        x, y, z = voxel_pos[:3]
        pos2d = search_region3d.project_to_2d(
            (x*res, y*res, z*res), search_region2d)

        if pos2d not in fov_cells:
            fov_cells[pos2d] = 'free'
        if voxel.label != Voxel.FREE:
            if objid is not None:
                if voxel.label == objid:
                    fov_cells[pos2d] = voxel.label
            else:
                fov_cells[pos2d] = voxel.label
    fov_cells = {(k, v) for k, v in fov_cells.items()}
    return fov_cells


### DEPRECATED ###
try:
    from .common import SloopMosAgent
    class SloopMosAgentBasic2D(SloopMosAgent):
        def _init_oopomdp(self, init_robot_pose_dist=None, init_object_beliefs=None):
            if init_robot_pose_dist is None:
                raise ValueError("To instantiate MosAgent, initial robot pose distribution is required.")

            mos_agent = MosAgentBasic2D(self.agent_config,
                                        self.search_region,
                                        init_robot_pose_dist=init_robot_pose_dist,
                                        init_object_beliefs=init_object_beliefs)
            return (mos_agent.belief,
                    mos_agent.policy_model,
                    mos_agent.transition_model,
                    mos_agent.observation_model,
                    mos_agent.reward_model)
except ImportError as ex:
    logging.error("Failed to import SloopMosAgent (basic2d): {}".format(ex))
