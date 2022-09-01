import math
import random
from pomdp_py import Gaussian
from sloop_object_search.utils.math import fround, euclidean_dist
from ..domain.observation import ObjectDetection, Voxel, ObjectVoxel
from .observation_model import ObjectDetectionModel
from .sensors import FanSensor, FrustumCamera
from .octree_belief import octree


################# Models based on FrustumCamera ############
class FrustumModel(ObjectDetectionModel):
    def __init__(self, objid, frustum_params,
                 quality_params, **kwargs):
        self.frustum_params = frustum_params
        self.quality_params = quality_params
        self._kwargs = kwargs
        self.__dict__.update(kwargs)
        super().__init__(objid)

    def copy(self):
        return self.__class__(self.objid,
                              self.frustum_params,
                              self.quality_params,
                              **self._kwargs)

    @property
    def observation_class(self):
        return ObjectDetection


class FrustumVoxelAlphaBeta(FrustumModel):
    """The alpha-beta model in MOS 3D

    This models Pr(o_i | s_i', a) where o_i is {(v,d(v)) for v in V_i}.

    This observation model is designed for planning, and makes the assumption
    that the object is contained within one voxel. This basically corresponds to
    VoxelObservationModel in 3D-MOS.
    """
    def __init__(self, objid, frustum_params, quality_params):
        """
        Args:
            objid (int) object id to detect
            quality_params; (alpha, beta) or (alpha, beta, gamma)
                detection_prob is essentially true positive rate.
        """
        super().__init__(objid, frustum_params,
                         quality_params)
        # by default, the camera looks at -z, as defined.
        self.sensor = FrustumCamera(**frustum_params)
        self.params = quality_params

    @property
    def alpha(self):
        return self.params[0]

    @property
    def beta(self):
        return self.params[1]

    @property
    def gamma(self):
        if len(self.params) == 3:
            return self.params[2]
        else:
            return octree.DEFAULT_VAL

    def sample(self, si, srobot, return_event=False):
        voxel = ObjectVoxel(self.objid, Voxel.NO_POSE, Voxel.UNKNOWN)
        if srobot.in_range(self.sensor, si):
            voxel = ObjectVoxel(self.objid, si.loc, Voxel.UNKNOWN)
            if FrustumCamera.sensor_functioning(
                    self.alpha, self.beta):
                voxel.label = si.id
            else:
                voxel.label = Voxel.OTHER
        return voxel

    def probability(self, zi, si, srobot):
        raise ValueError("per-voxel observation model is not"
                         "meant for belief update.")
