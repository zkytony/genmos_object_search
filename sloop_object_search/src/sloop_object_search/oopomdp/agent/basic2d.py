# basic2d: 2D search agent with primitive action space.
import pomdp_py
from tqdm import tqdm
from ..models.search_region import SearchRegion2D
from ..models.transition_model import RobotTransBasic2D
from ..models.policy_model import PolicyModelBasic2D
from ..models import belief
from ..domain.observation import JointObservation, ObjectDetection
from .common import MosAgent, SloopMosAgent, init_object_transition_models, init_primitive_movements
from sloop_object_search.utils import open3d_utils

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

    def _update_object_beliefs(self, observation, action=None, debug=False, **kwargs):
        assert isinstance(observation, JointObservation)
        if not self.robot_id in observation:
            raise ValueError("requires knowing robot pose corresponding"\
                             " to the object detections.")

        robot_pose_est = observation.z(self.robot_id).pose_est
        robot_pose = robot_pose_est.pose
        if robot_pose_est.is_3d:
            robot_pose = observation.z(self.robot_id).pose_est.to_2d().pose

        for objid in observation:
            if objid == self.robot_id:
                continue
            zobj = observation.z(objid)
            if not isinstance(zobj, ObjectDetection):
                raise NotImplementedError(f"Unable to handle object observation of type {type(zobj)}")
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
                           debug=True, discrete=True):
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
