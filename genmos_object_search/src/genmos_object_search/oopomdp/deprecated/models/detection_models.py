import math
import random
from pomdp_py import Gaussian
from genmos_object_search.utils.math import fround, euclidean_dist
from ...domain.observation import ObjectDetection
from ...models.observation_model import ObjectDetectionModel
from ...models.sensors import FanSensor

################# Models based on FanSensor ############
class FanModel(ObjectDetectionModel):
    def __init__(self, objid, fan_params,
                 quality_params, round_to="int", **kwargs):
        self.fan_params = fan_params
        self.quality_params = quality_params
        self._round_to = round_to
        self._kwargs = kwargs
        self.__dict__.update(kwargs)
        super().__init__(objid)

    def copy(self):
        return self.__class__(self.objid,
                              self.fan_params,
                              self.quality_params,
                              round_to=self._round_to,
                              **self._kwargs)

    @property
    def observation_class(self):
        return ObjectDetection


class FanModelYoonseon(FanModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')
    Yoonseon's model proposed in the OOPOMDP paper;

    Pros: parameterization is simplistic;
          simulates both false positive and false negatives.

    Cons: false positive assumption is unrealistic;
          False positive and false negative rates are the same;
          parameter values is too harsh and don't have good semantics
          (e.g. epsilon=0.9 is a pretty bad sensor already).
    """
    def __init__(self, objid, fan_params,
                 quality_params, round_to="int"):
        """
        objid: the class detected by this model
        fan_params: initialization params for FanSensor
        quality_params: a (sigma, epsilon) tuple
            See the definition of the 2D MOS sensor model in
            the OO-POMDP paper.
        round_to: Round the sampled observation locations to,
            either a float, 'int', or None
        """
        super().__init__(objid, fan_params,
                         quality_params, round_to="int")
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params

    def _compute_params(self, object_in_sensing_region, epsilon):
        if object_in_sensing_region:
            # Object is in the sensing region
            alpha = epsilon
            beta = (1.0 - epsilon) / 2.0
            gamma = (1.0 - epsilon) / 2.0
        else:
            # Object is not in the sensing region.
            alpha = (1.0 - epsilon) / 2.0
            beta = (1.0 - epsilon) / 2.0
            gamma = epsilon
        return alpha, beta, gamma

    def probability(self, zi, si, srobot):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        sigma, epsilon = self.params
        alpha, beta, gamma = self._compute_params(
            srobot.in_range(self.sensor, si), epsilon)

        # Requires Python >= 3.6
        prob = 1e-12
        # Event A:
        # object in sensing region and observation comes from object i
        if zi.loc is None:
            # Even though event A occurred, the observation is NULL.
            # This has 0.0 probability.
            prob += 0.0 * alpha
        else:
            gaussian = Gaussian(list(si),
                                [[sigma**2, 0],
                                 [0, sigma**2]])
            prob += gaussian[zi.loc] * alpha

        # Event B
        prob += (1.0 / self.sensor.sensor_region_size) * beta

        # Event C
        pr_c = 1.0 if zi.loc is None else 0.0  # indicator zi == NULL
        prob += pr_c * gamma
        return prob

    def sample(self, si, srobot, return_event=False):
        sigma, epsilon = self.params
        alpha, beta, gamma = self._compute_params(
            srobot.in_range(self.sensor, si), epsilon)

        event_occured = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        if event_occured == "A":
            gaussian = Gaussian(list(si.loc),
                                [[sigma**2, 0],
                                 [0, sigma**2]])
            # Needs to discretize otherwise MCTS tree cannot handle this.
            loc = fround("int", gaussian.random())

        elif event_occured == "B":
            # Sample from field of view
            loc_cont = self.sensor.uniform_sample_sensor_region(srobot["pose"])
            loc = fround(self._round_to, loc_cont)
        else:  # event == C
            loc = None
        zi = ObjectDetection(si.objid, loc)
        if return_event:
            return zi, event_occured
        else:
            return zi


class FanModelNoFP(FanModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')

    Model without involving false positives

    Pros: semantic parameter;
    Cons: no false positives modeled
    """
    def __init__(self, objid, fan_params, quality_params, round_to="int"):
        """
        objid: the class detected by this model
        fan_params: initialization params for FanSensor
        quality_params: (detection probability, sigma)
        round_to: Round the sampled observation locations to,
            either a float, 'int', or None
        """
        super().__init__(objid, fan_params,
                         quality_params, round_to="int")
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params  # calling it self.params to have consistent interface

    @property
    def detection_prob(self):
        return self.params[0]

    @property
    def sigma(self):
        return self.params[1]

    def probability(self, zi, si, srobot):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            if zi.loc is None:
                # false negative
                return 1.0 - self.detection_prob
            else:
                # True positive; gaussian centered at object loc
                gaussian = Gaussian(list(si.loc),
                                    [[self.sigma**2, 0],
                                     [0, self.sigma**2]])
                return self.detection_prob * gaussian[zi.loc]
        else:
            if zi.loc is None:
                # True negative; we are not modeling false positives
                return 1.0
            else:
                return 1e-12


    def sample(self, si, srobot, return_event=False):
        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            if random.uniform(0,1) <= self.detection_prob:
                # sample according to gaussian
                gaussian = Gaussian(list(si.loc),
                                    [[self.sigma**2, 0],
                                     [0, self.sigma**2]])
                loc = tuple(fround(self._round_to, gaussian.random()))
                zi = ObjectDetection(si.id, loc)
                event = "detected"

            else:
                zi = ObjectDetection(si.id, None)
                event = "missed"
        else:
            zi = ObjectDetection(si.id, None)
            event = "out_of_range"
        if return_event:
            return zi, event
        else:
            return zi


class FanModelSimpleFP(FanModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')

    Considers false positive rate during belief update, but not when
    sampling. You could say there is an implicit variable in the distribution
    that indicates whether it is evaluating a probability or sampling.
    It is therefore valid to define such a distribution.

    Pros: semantic parameter;
    Cons: false positive is not sampled & out of context
    """
    def __init__(self, objid, fan_params, quality_params, round_to="int"):
        """
        Args:
            objid (int) object id to detect
            quality_params; (detection_prob, false_pos_rate, sigma);
                detection_prob is essentially true positive rate.
        """
        super().__init__(objid, fan_params,
                         quality_params, round_to="int")
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params

    @property
    def detection_prob(self):
        return self.params[0]

    @property
    def sigma(self):
        return self.params[2]

    @property
    def false_pos_rate(self):
        return self.params[1]

    def probability(self, zi, si, srobot):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            if zi.loc is None:
                # false negative
                return 1.0 - self.detection_prob
            else:
                if not srobot.loc_in_range(self.sensor, zi.loc):
                    # the robot would not have received such a positive observation,
                    # because it is outside of the FOV. It is treatd as a false positive,
                    # that comes uniformly likely outside of the FOV. We estimate the
                    # size of the world here as the fan sensor isn't provided the grid map.
                    return self.false_pos_rate / (100 - (self.sensor.sensor_region_size))

                # determin if zi is true positive or true negative.  If it is within
                # 3*sigma range from si.loc then it is a true positvie. Otherwise,
                # it is a false positive. The probability of the false positive is
                # uniform within the sensor field range.
                if euclidean_dist(zi.loc, si.loc) > 3*self.sigma:
                    # false positive
                    return self.false_pos_rate / self.sensor.sensor_region_size
                else:
                    # true positive
                    # True positive; gaussian centered at object loc
                    gaussian = Gaussian(list(si.loc),
                                        [[self.sigma**2, 0],
                                         [0, self.sigma**2]])
                    return self.detection_prob * gaussian[zi.loc]
        else:
            if zi.loc is None:
                # True negative;
                return 1.0 - self.false_pos_rate
            else:
                return self.false_pos_rate / self.sensor.sensor_region_size


    def sample(self, si, srobot, return_event=False):
        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            if random.uniform(0,1) <= self.detection_prob:
                # sample according to gaussian
                gaussian = Gaussian(list(si.loc),
                                    [[self.sigma**2, 0],
                                     [0, self.sigma**2]])
                loc = tuple(fround(self._round_to, gaussian.random()))
                zi = ObjectDetection(si.id, loc)
                event = "detected"

            else:
                zi = ObjectDetection(si.id, None)
                event = "missed"
        else:
            zi = ObjectDetection(si.id, None)
            event = "out_of_range"
        if return_event:
            return zi, event
        else:
            return zi


class FanModelFarRange(FanModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')

    Considers false positive rate during belief update, but not when
    sampling. You could say there is an implicit variable in the distribution
    that indicates whether it is evaluating a probability or sampling.
    It is therefore valid to define such a distribution.

    This model assumes objects could be detected infinitely far
    away, but the probability of detection falls off exponentially
    as the detection is farther away from the average detection range.
    If it is within the detection range, then we say it is expected
    to be detected.

    Pros: semantic parameter;
    Cons: false positive is not sampled & out of context
    """
    def __init__(self, objid, fan_params, quality_params,
                 round_to="int", max_range_limit=100):
        """
        Args:
            objid (int) object id to detect
            fan_params (dict): parameters for FanSensor; Will ignore the setting of max_range;
                The 'max_range' parameter in this dictionary, however, will be regarded
                as the 'average detection distance'
            quality_params; (detection_prob, false_pos_rate, sigma);
                detection_prob is essentially true positive rate.
        """
        super().__init__(objid, fan_params,
                         quality_params,
                         round_to="int",
                         max_range_limit=max_range_limit)

        fan_params['max_range'] = max_range_limit
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params

    @property
    def detection_prob(self):
        return self.params[0]

    @property
    def sigma(self):
        return self.params[2]

    @property
    def false_pos_rate(self):
        return self.params[1]

    def probability(self, zi, si, srobot):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        # the observation is positive. Now, the probability
        # of this detection is subject to the distance of
        # the detection.
        if zi.loc is not None:
            distance = euclidean_dist(zi.loc, srobot.loc)
            if distance <= self.sensor.mean_range:
                distance_weight = 1.0
            else:
                distance_weight = math.exp(-(distance - self.sensor.mean_range)**2)

        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            # This is still important, for angular range.

            if zi.loc is None:
                # false negative
                return 1.0 - self.detection_prob

            else:
                # We will regard the detection as a true positive if it is
                # within 3*sigma from the object's true location given in si.
                # Otherwise, it is a false positiv  The probability of the false positive is
                # uniform within the sensor field range.
                if euclidean_dist(zi.loc, si.loc) > 3*self.sigma:
                    # false positive
                    return distance_weight * self.false_pos_rate / self.sensor.sensor_region_size
                else:
                    # true positive
                    # True positive; gaussian centered at object loc
                    gaussian = Gaussian(list(si.loc),
                                        [[self.sigma**2, 0],
                                         [0, self.sigma**2]])
                    return distance_weight * self.detection_prob * gaussian[zi.loc]
        else:
            # Not within angular range
            if zi.loc is None:
                # True negative;
                return 1.0 - self.false_pos_rate

            else:
                return distance_weight * self.false_pos_rate / self.sensor.sensor_region_size


    def sample(self, si, srobot, return_event=False):
        in_range = srobot.in_range(self.sensor, si)

        # si is detectable if it is also within the average detection
        # distance. Otherwise, we will not consider it detectable; This is a
        # simpler model than in probability, but it faciliates behavior to drive
        # the robot closer to where it believes the object is
        distance = euclidean_dist(si.loc, srobot.loc)
        if distance > self.sensor.mean_range:
            in_range = False

        if in_range:
            if random.uniform(0,1) <= self.detection_prob:
                # sample according to gaussian
                gaussian = Gaussian(list(si.loc),
                                    [[self.sigma**2, 0],
                                     [0, self.sigma**2]])
                loc = tuple(fround(self._round_to, gaussian.random()))
                zi = ObjectDetection(si.id, loc)
                event = "detected"

            else:
                zi = ObjectDetection(si.id, None)
                event = "missed"
        else:
            zi = ObjectDetection(si.id, None)
            event = "out_of_range"

        if return_event:
            return zi, event
        else:
            return zi


class FanModelSimpleFPLabelOnly(FanModel):
    """Intended for 2D-level observation; Pr(zi | si, srobot')

    The difference between this model and FanModelSimpleFP is that
    this model does not care whether the detected object's location
    matches the object location in the state. As long as, if the object
    is within the field of view, and when the detection is non-null,
    that counts as a positive detection. This is useful when there is
    a lot of noise related to object detection's locations.

    Pros: semantic parameter;
    Cons: false positive is not sampled & out of context
    """
    def __init__(self, objid, fan_params, quality_params, round_to="int"):
        """
        Args:
            objid (int) object id to detect
            quality_params; (detection_prob, false_pos_rate, sigma);
                detection_prob is essentially true positive rate.
        """
        super().__init__(objid, fan_params,
                         quality_params, round_to="int")
        self.sensor = FanSensor(**fan_params)
        self.params = quality_params

    @property
    def detection_prob(self):
        return self.params[0]

    @property
    def false_pos_rate(self):
        return self.params[1]

    def probability(self, zi, si, srobot):
        """
        zi (LocDetection)
        si (HLObjectstate)
        srobot (HLObjectstate)
        """
        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            if zi.loc is None:
                # false negative
                return 1.0 - self.detection_prob
            else:
                # true positive
                return self.detection_prob
        else:
            if zi.loc is None:
                # True negative;
                return 1.0 - self.false_pos_rate
            else:
                return self.false_pos_rate

    def sample(self, si, srobot, return_event=False):
        in_range = srobot.in_range(self.sensor, si)
        if in_range:
            if random.uniform(0,1) <= self.detection_prob:
                zi = ObjectDetection(si.id, ObjectDetection.NO_POSE)
                event = "detected"
            else:
                zi = ObjectDetection(si.id, None)
                event = "missed"
        else:
            zi = ObjectDetection(si.id, None)
            event = "out_of_range"

        if return_event:
            return zi, event
        else:
            return zi
