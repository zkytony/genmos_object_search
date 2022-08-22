# Copyright 2022 Kaiyu Zheng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as scipyR
import scipy.stats as stats
import math
import scipy.stats
import pandas as pd


# Operations
def remap(oldval, oldmin, oldmax, newmin, newmax, enforce=False):
    newval = (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
    if enforce:
        return min(max(newval, newmin), newmax)
    else:
        return newval

def closest(values, query):
    """Returns the entry in `values` that is
    closest to `query` in terms of absolute value difference."""
    return min(values, key=lambda v: abs(v-query))

def normalize_angles(angles):
    """Returns array-like of angles within 0 to 360 degrees"""
    return type(angles)(map(lambda x: x % 360, angles))

def roundany(x, base):
    """
    rounds the number x (integer or float) to
    the closest number that increments by `base`.
    """
    return base * round(x / base)

def floorany(x, base):
    return base * math.floor(x / base)

def clip(x, minval, maxval):
    return min(maxval, max(x, minval))

def diff(rang):
    return rang[1] - rang[0]

def in_range(x, rang):
    return x >= rang[0] and x < rang[1]

def in_range_inclusive(x, rang):
    return x >= rang[0] and x <= rang[1]

def in_region(p, ranges):
    """check if every element in p is within given ranges;
    Note that the size of 'ranges' could be larger or equal
    than the size of 'p'"""
    return all(in_range(p[i], ranges[i])
               for i in range(len(p)))

def in_box3d_center(p, box):
    """Returns true if point 'p' is inside the 3D box 'box'.
    The box is represented as a tuple (center, w, l, h), where
    center is the center point of the box (cx,cy,cz), and lx, ly, lz
    are dimensions along x, y, z axes, respectively."""
    if len(p) != 3:
        raise ValueError("Requires point to be 3D")
    center, lx, ly, lz = box
    px, py, pz = p
    cx, cy, cz = center
    return abs(px - cx) <= lx/2\
        and abs(py - cy) <= ly/2\
        and abs(pz - cz) <= lz/2

def in_box3d_origin(p, box):
    """Returns true if point 'p' is inside the 3D box 'box'.
    The box is represented as a tuple (origin, w, l, h), where
    center is the center point of the box (ox,oy,oz), and lx, ly, lz
    are dimensions along x, y, z axes, respectively."""
    if len(p) != 3:
        raise ValueError("Requires point to be 3D")
    center, lx, ly, lz = box
    px, py, pz = p
    ox, oy, oz = center
    return abs(px - ox) <= lx\
        and abs(py - oy) <= ly\
        and abs(pz - oz) <= lz

def boxes_overlap3d_origin(box1, box2):
    """Return True if the two origin-based boxes overlap
    https://stackoverflow.com/a/53488289/2893053"""
    origin1, w1, l1, h1 = box1
    origin2, w2, l2, h2 = box2
    box1_min_x = origin1[0]
    box1_max_x = origin1[0] + w1
    box1_min_y = origin1[1]
    box1_max_y = origin1[1] + l1
    box1_min_z = origin1[2]
    box1_max_z = origin1[2] + h1
    box2_min_x = origin2[0]
    box2_max_x = origin2[0] + w1
    box2_min_y = origin2[1]
    box2_max_y = origin2[1] + l1
    box2_min_z = origin2[2]
    box2_max_z = origin2[2] + h1
    return (box1_min_x < box2_max_x)\
        and (box1_max_x > box2_min_x)\
        and (box1_min_y < box2_max_y)\
        and (box1_max_y > box2_min_y)\
        and (box1_min_z < box2_max_z)\
        and (box1_max_z > box2_max_z)


def sample_in_box3d_origin(box):
    """Given a box represented as a tuple (origin, w, l, h),
    returns a point sampled from within the box"""
    origin, w, l, h = box
    dx = random.uniform(0, w)
    dy = random.uniform(0, l)
    dz = random.uniform(0, h)
    return (origin[0] + dx,
            origin[1] + dy,
            origin[2] + dz)

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True

def identity(a, b, epsilon=1e-9):
    if a == b:
        return 1 - epsilon
    else:
        return epsilon

######## Conversions
def to_radians(th):
    return th*np.pi / 180

def to_rad(th):
    return th*np.pi / 180

def to_degrees(th):
    return th*180 / np.pi

def to_deg(th):
    return th*180 / np.pi

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


########## Transform; all input angles are degrees
def R_x(th):
    th = to_rad(th)
    return np.array([
        1, 0, 0, 0,
        0, np.cos(th), -np.sin(th), 0,
        0, np.sin(th), np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_y(th):
    th = to_rad(th)
    return np.array([
        np.cos(th), 0, np.sin(th), 0,
        0, 1, 0, 0,
        -np.sin(th), 0, np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_z(th):
    th = to_rad(th)
    return np.array([
        np.cos(th), -np.sin(th), 0, 0,
        np.sin(th), np.cos(th), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R2d(th):
    th = to_rad(th)
    return np.array([
        np.cos(th), -np.sin(th),
        np.sin(th), np.cos(th)
    ]).reshape(2,2)

def R_between(v1, v2):
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Only applicable to 3D vectors!")
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    I = np.identity(3)

    vX = np.array([
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0
    ]).reshape(3,3)
    R = I + vX + np.matmul(vX,vX) * ((1-c)/(s**2))
    return R

def R_euler(thx, thy, thz, affine=False, order='xyz'):
    """
    Obtain the rotation matrix of Rz(thx) * Ry(thy) * Rx(thz); euler angles
    """
    R = scipyR.from_euler(order, [thx, thy, thz], degrees=True)
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_matrix()
        aR[3,3] = 1
        R = aR
    return R

def R_quat(x, y, z, w, affine=False):
    R = scipyR.from_quat([x,y,z,w])
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_matrix()
        aR[3,3] = 1
        R = aR
    return R

def R_to_euler(R, order='xyz'):
    """
    Obtain the thx,thy,thz angles that result in the rotation matrix Rz(thx) * Ry(thy) * Rx(thz)
    Reference: http://planning.cs.uiuc.edu/node103.html
    """
    return R.as_euler(order, degrees=True)

def R_to_quat(R):
    return R.as_quat()

def euler_to_quat(thx, thy, thz, order='xyz'):
    return scipyR.from_euler(order, [thx, thy, thz], degrees=True).as_quat()

def quat_to_euler(x, y, z, w, order='xyz'):
    return scipyR.from_quat([x,y,z,w]).as_euler(order, degrees=True)

def T(dx, dy, dz):
    return np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]).reshape(4,4)

def vec(p1, p2):
    """ vector from p1 to p2 """
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    if type(p2) != np.ndarray:
        p2 = np.array(p2)
    return p2 - p1

def proj(vec1, vec2, scalar=False):
    # Project vec1 onto vec2. Returns a vector in the direction of vec2.
    scale = np.dot(vec1, vec2) / np.linalg.norm(vec2)
    if scalar:
        return scale
    else:
        return vec2 * scale

def angle_between(v1, v2):
    """returns the angle between two vectors. The result is in degrees
    reference: https://stackoverflow.com/a/13849249/2893053"""
    return to_deg(np.arccos(
        np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)))

# Probability
def sep_spatial_sample(candidates, sep, num_samples,
                       sample_func=None, rnd=random):
    """Draws samples from candidates,
    such that the samples are minimally of euclidean distance
    `sep` apart. Note that this will attempt to
    draw `num_samples` samples but is not guaranteed
    to return this number of samples.

    You can optionally supply a sample_func
    that takes in the candidates and return
    a sample. If not provided, will draw uniformly.

    The samples will not be at duplicate locations."""
    samples = set()
    for _ in range(num_samples):
        if sample_func is None:
            s = rnd.sample(candidates, 1)[0]
        else:
            s = sample_func(candidates)

        if len(samples) > 0:
            closest = min(samples,
                          key=lambda c: euclidean_dist(s, c))
            if euclidean_dist(closest, s) >= sep:
                samples.add(s)
        else:
            samples.add(s)
    return samples

def normalize(ss):
    """
    ss (dict or array-like):
        If dict, maps from key to float; If array-like, all floats.
    Returns:
        a new object (of the same kind as input) with normalized entries
    """
    if type(ss) == dict:
        total = sum(ss.values())
        return {k:ss[k]/total for k in ss}
    elif type(ss) == np.ndarray:
        return ss/np.linalg.norm(ss)
    else:
        total = sum(ss)
        return type(ss)(ss[i]/total for i in range(len(ss)))

def uniform(size, ranges):
    return tuple(random.randrange(ranges[i][0], ranges[i][1])
                 for i in range(size))

def indicator(cond, epsilon=0.0):
    return 1.0 - epsilon if cond else epsilon

def normalize_log_prob(likelihoods):
    """Given an np.ndarray of log values, take the values out of the log space,
    and normalize them so that they sum up to 1"""
    normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                        (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return normalized

## Numbers
def roundany(x, base):
    """
    rounds the number x (integer or float) to
    the closest number that increments by `base`.
    """
    n_digits = math.log10(base)
    if n_digits.is_integer():
        # using built-in round function avoids numerical instability
        # for the case of rounding to 0.00...001
        return round(x, abs(int(n_digits)))
    else:
        return base * round(x / base)

def fround(round_to, loc_cont):
    """My own 'float'-rounding (hence fround) method.

    round_to can be 'int', 'int-' or any float,
    and will output a value that is the result of
    rounding `loc_cont` to integer, or the `round_to` float;
    (latter uses roundany).

    If 'int-', will floor `loc_cont`
    """
    if round_to == "int":
        if hasattr(loc_cont, "__len__"):
            return tuple(map(lambda x: int(round(x)), loc_cont))
        else:
            return int(round(loc_cont))
    elif round_to == "int-":
        if hasattr(loc_cont, "__len__"):
            return tuple(map(lambda x: int(math.floor(x)), loc_cont))
        else:
            return int(math.floor(loc_cont))
    elif type(round_to) == float:
        if hasattr(loc_cont, "__len__"):
            return tuple(map(lambda x: roundany(x, round_to),
                             loc_cont))
        else:
            return roundany(loc_cont, round_to)
    else:
        raise ValueError(f"unrecognized option for 'round_to': {round_to}")

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True

## Geometry
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def eucdist_multi(points, p):
    """returns euclidean distance between an array of points and a point p"""
    return np.linalg.norm(points - p, axis=1)

def in_square(point, center, size):
    """Returns true if point is in a square
    with length 'size' centered at 'center'"""
    return all(abs(point[i] - center[i]) <= size/2
               for i in range(len(point)))

def in_square_multi(points, center, size):
    """Given numpy array of 'points' (N,2), center (1,2) and size, return
    a mask where True means the corresponding point is within
    a square centered at 'center' with size 'size'"""
    return np.all(np.abs(points - np.asarray(center)) <= size/2, axis=1)

def intersect(seg1, seg2):
    """seg1 and seg2 are two line segments each represented by
    the end points (x,y). The algorithm comes from
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect"""
    # Represent each segment (p,p+r) where r = vector of the line segment
    seg1 = np.asarray(seg1)
    p, r = seg1[0], seg1[1]-seg1[0]

    seg2 = np.asarray(seg2)
    q, s = seg2[0], seg2[1]-seg2[0]

    r_cross_s = np.cross(r, s)
    if r_cross_s != 0:
        t = np.cross(q-p, s) / r_cross_s
        u = np.cross(q-p, r) / r_cross_s
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Two lines meet at point
            return True
        else:
            # Are not parallel and not intersecting
            return False
    else:
        if np.cross(q-p, r) == 0:
            # colinear
            t0 = np.dot((q - p), r) / np.dot(r, r)
            t1 = t0 + np.dot(s, r) / np.dot(r, r)
            if t0 <= 0 <= t1 or t0 <= 1 <= t1:
                # colinear and overlapping
                return True
            else:
                # colinear and disjoint
                return False
        else:
            # two lines are parallel and non intersecting
            return False

def overlap(seg1, seg2):
    """returns true if line segments seg1 and 2 are
    colinear and overlapping"""
    seg1 = np.asarray(seg1)
    p, r = seg1[0], seg1[1]-seg1[0]

    seg2 = np.asarray(seg2)
    q, s = seg2[0], seg2[1]-seg2[0]

    r_cross_s = np.cross(r, s)
    if r_cross_s == 0:
        if np.cross(q-p, r) == 0:
            # colinear
            t0 = np.dot((q - p), r) / np.dot(r, r)
            t1 = t0 + np.dot(s, r) / np.dot(r, r)
            if t0 <= 0 <= t1 or t0 <= 1 <= t1:
                # colinear and overlapping
                return True
    return False

def law_of_cos(a, b, angle):
    """Given length of edges a and b and angle (degrees)
    in between, return the length of the edge opposite to the angle"""
    return math.sqrt(a**2 + b**2 - 2*a*b*math.cos(to_rad(angle)))

def inverse_law_of_cos(a, b, c):
    """Given three edges, a, b, c, figure out
    the angle between a and b (i.e. opposite of c), in degrees"""
    costh = (a**2 + b**2 - c**2) / (2*a*b)
    return to_deg(math.acos(costh))

## Statistics
# confidence interval
def ci_normal(series, confidence_interval=0.95):
    series = np.asarray(series)
    tscore = stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)
    y_error = stats.sem(series)
    ci = y_error * tscore
    return ci

def mean_ci_normal(series, confidence_interval=0.95):
    ### CODE BY CLEMENT at LIS ###
    """Confidence interval for normal distribution with unknown mean and variance.

    Interpretation:
    An easy way to remember the relationship between a 95%
    confidence interval and a p-value of 0.05 is to think of the confidence interval
    as arms that "embrace" values that are consistent with the data. If the null
    value is "embraced", then it is certainly not rejected, i.e. the p-value must be
    greater than 0.05 (not statistically significant) if the null value is within
    the interval. However, if the 95% CI excludes the null value, then the null
    hypothesis has been rejected, and the p-value must be < 0.05.
    """
    series = np.asarray(series)
    # this is the "percentage point function" which is the inverse of a cdf
    # divide by 2 because we are making a two-tailed claim
    tscore = scipy.stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)

    y_mean = np.mean(series)
    y_error = scipy.stats.sem(series)

    half_width = y_error * tscore
    return y_mean, half_width

def perplexity(p, base=2):
    """Measures how well a probability model predicts a sample (from the same
    distribution). On Wikipedia, given a probability distribution p(x), this
    distribution has a notion of "perplexity" defined as:

        2 ^ (- sum_x p(x) * log_2 p(x))

    The exponent is also called the "entropy" of the distribution.
    These two words, "perplexity" amd "entropy" are designed
    to be vague by Shannon himself (according to Tomas L-Perez).

    The interpretation of entropy is "level of information",
    or "level of uncertainty" inherent in the distribution.
    Higher entropy indicates higher "randomness" of the
    distribution. You can actually observe the scale of

    "p(x) * log_2 p(x)" in a graph tool (e.g. Desmos).

    I made the probability distribution of x to be P(x) = 1/2 * (sin(x)+1).
    (It's not normalized, but it's still a distribution.)
    You observe that when P(x) approaches 1, the value of p(x) * log_2 p(x)
    approaches zero. The farther P(x) is away from 1, the more negative
    is the value of p(x) * log_2 p(x). You can understand it has,
    if p(x) * log_2 p(x) --> 0, then the value of x here does not
    contribute much to the uncertainty since P(x) --> 1.
    Thus, if we sum up these quantities

    "sum_x p(x) * log_2 p(x)"
    we can understand it as: the lower this sum, the more uncertainty there is.
    If you take the negation, then it becomes the higher this quantity,
    the more the uncertainty, which is the definition of entropy.

    Why log base 2? It's "for convenience and intuition" (by Shannon) In fact
    you can do other bases, like 10, or e.

    Notice how KL-divergence is defined as:

    "-sum_x p(x) * log_2 ( p(x) / q(x) )"

    The only difference is there's another distribution q(x). It measures
    how "different" two distributions are. KL divergence of 0 means identical.

    How do you use perplexity to compare two distributions? You compute the
    perplexity of both.

    Also refer to: https://www.cs.rochester.edu/u/james/CSC248/Lec6.pdf

    Parameters:
        p: A sequence of probabilities
    """
    H = scipy.stats.entropy(p, base=base)
    return base**H

def kl_divergence(p, q, base=2):
    return scipy.stats.entropy(p, q, base=base)

def normal_pdf_2d(point, variance, domain, normalize=True):
    """
    returns a dictionary that maps a value in domain to a probability
    such that the probability distribution is a 2d gaussian with mean
    at the given point and given variance.
    """
    dist = {}
    total_prob = 0.0
    for val in domain:
        prob = scipy.stats.multivariate_normal.pdf(np.array(val),
                                                   np.array(point),
                                                   np.array(variance))
        dist[val] = prob
        total_prob += prob
    if normalize:
        for val in dist:
            dist[val] /= total_prob
    return dist

def dists_to_seqs(dists, avoid_zero=True):
    """Convert dictionary distributions to seqs (lists) such
    that the elements at the same index in the seqs correspond
    to the same key in the dictionary"""
    seqs = [[] for i in range(len(dists))]
    vals = []
    d0 = dists[0]
    for val in d0:
        for i, di in enumerate(dists):
            if val not in di:
                raise ValueError("Value %s is in one distribution but not another" % (str(val)))
            if avoid_zero:
                prob = max(1e-12, di[val])
            else:
                prob = di[val]
            seqs[i].append(prob)
        vals.append(val)
    return seqs, vals

def compute_mean_ci(results):
    """Given `results`, a dictionary mapping "result_type" to a list of values
    for this result_type, compute the mean and confidence intervals for each
    of the result type. It will add a __summary__ key to the given dictionary.x"""
    results["__summary__"] = {}
    for restype in results:
        if restype.startswith("__"):
            continue
        mean, ci = mean_ci_normal(results[restype], confidence_interval=0.95)
        results["__summary__"][restype] = {
            "mean": mean,
            "ci-95": ci,
            "size": len(results[restype])
        }
    return results

def entropy(p, base=2):
    """
    Parameters:
        p: A sequence of probabilities
    """
    return scipy.stats.entropy(p, base=base)

def indicies2d(m, n):
    # reference: https://stackoverflow.com/a/44230705/2893053
    return np.indices((m,n)).transpose(1,2,0)


def tind_test(sample1, sample2):
    """Performs a two-sample independent t-test.  Note that in statistics, a sample
    is a set of individuals (observations) or objects collected or selected from
    a statistical population by a defined procedure.

    references:
    https://www.reneshbedre.com/blog/ttest.html#two-sample-t-test-unpaired-or-independent-t-test.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

    Formula:

        t = (m1 - m2)  / sqrt ( s^2 (1/n1 + 1/n2) )

    where m1, m2 are the means of two independent samples, and s^2 is the "pooled"
    sample variance, calculated as:

        s^2 = [ (n1-1)s1^2 + (n2-1)s2^2 ] / (n1 + n2 - 2)

    and n1, n2 are the sample sizes

    Note: Independent samples are samples that are selected randomly so that its
    observations do not depend on the values other observations.

    Args:
        sample1 (list or numpy array)
        sample2 (list or numpy array)
    Returns:
        (tstatistic, pvalue)
    """
    res = stats.ttest_ind(a=sample1, b=sample2, equal_var=True)
    return res.statistic, res.pvalue


def pval2str(pval):
    """Converts p value to ns, *, **, etc.
    Uses the common convention."""
    if math.isnan(pval):
        return "NA"
    if pval > 0.05:
        return "ns"
    else:
        if pval <= 0.0001:
            return "****"
        elif pval <= 0.001:
            return "***"
        elif pval <= 0.01:
            return "**"
        else:
            return "*"

def wilcoxon_test(sample1, sample2):
    """This is a nonparametric test (i.e. no need to compute statistical
    parameter from the sample like mean).

    From Wikipedia:
    When applied to test the location of a set of samples, it serves the same
    purpose as the one-sample Student's t-test.

    On a set of matched samples, it is a paired difference test like the paired
    Student's t-test

    Unlike Student's t-test, the Wilcoxon signed-rank test does not assume that
    the differences between paired samples are normally distributed.

    On a wide variety of data sets, it has greater statistical power than
    Student's t-test and is more likely to produce a statistically significant
    result. The cost of this applicability is that it has less statistical power
    than Student's t-test when the data are normally distributed.
    """
    def _all_zeros(sample):
        return all([abs(s) <= 1e-12 for s in sample])
    if _all_zeros(sample1) and _all_zeros(sample2):
        # the test cannot be performed; the two samples have no difference
        return float('nan'), float('nan')

    res = stats.wilcoxon(x=sample1, y=sample2)
    return res.statistic, res.pvalue


def test_significance_pairwise(results, sigstr=False, method="t_ind"):
    """
    Runs statistical significance tests for all pairwise combinations.
    Returns result as a table. Uses two-sample t-test. Assumes independent sample.

    Args:
        results (dict): Maps from method name to a list of values for the result.
        sigstr (bool): If True, then the table entries will be strings like *, **, ns etc;
             otherwise, they will be pvalues.
    Returns:
        pd.DataFrame: (X, Y) entry will be the statistical significance between
            method X and method Y.
    """
    method_names = list(results.keys())
    rows = []
    for meth1 in method_names:
        row = []
        for meth2 in method_names:
            if meth1 == meth2:
                row.append("-");
            else:
                if method == "t_ind":
                    _, pval = tind_test(results[meth1], results[meth2])
                elif method == "wilcoxon":
                    _, pval = wilcoxon_test(results[meth1], results[meth2])
                else:
                    raise ValueError("Unable to perform significance test {}".format(method))

                pvalstr = "{0:.4f}".format(pval)
                if sigstr:
                    pvalstr += " ({})".format(pval2str(pval))
                row.append(pvalstr)
        rows.append(row)
    df = pd.DataFrame(rows, method_names, method_names)
    return df
