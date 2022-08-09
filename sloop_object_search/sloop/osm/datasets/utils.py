import numpy as np
import math
from scipy import ndimage

# ALL img means map image.

def is_at_border(x, y, footprint):
    # If a point does not have all of its neighboring points
    # inside the footprint, then it is a border point.
    if (x,y) not in footprint:
        return False
    for dx in {-1,0,1}:
        for dy in {-1,0,1}:
            if (x+dx, y+dy) not in footprint:
                return True
    return False

# none ego centric
def make_img(mapinfo, map_name, landmarks,
             landmark_border=True, landmark_values={}):
    """If landmark border is true, then the polygon that surrounds
    the landmark will have a border with value -1"""
    arr = np.zeros(mapinfo.map_dims(map_name))
    for lm in landmarks:
        footprint = set(mapinfo.landmark_footprint(lm, map_name))
        if lm in landmark_values:
            val, border_val = landmark_values[lm]
        else:
            val = 100
            border_val = -100
        for x,y in footprint:
            if landmark_border and is_at_border(x, y, footprint):
                arr[x,y] = border_val
            else:
                arr[x,y] = val
    return arr

def make_context_img(mapinfo, map_name,
                     landmark, landmark_border=True,
                     dist_factor=2.0):
    """Given a single landmark, find the surrounding streets of that landmark,
    and make a image."""
    dims = mapinfo.xyrange(landmark, map_name)
    dist = max(dims) * dist_factor
    lmk_ctr = mapinfo.center_of_mass(landmark, map_name)
    lmks_around = set()
    lmk_vals = {}
    for cell in mapinfo.cell_to_landmark(map_name):
        if euclidean_dist(lmk_ctr, cell) < dist:
            lmk = mapinfo.landmark_at(map_name, cell)
            if lmk in mapinfo.streets(map_name):
                lmks_around.add(lmk)
                if lmk != landmark:
                    # Color streets differently (unless they are the landmark itself)
                    lmk_vals[lmk] = (-200, -200)
                else:
                    lmk_vals[lmk] = (100, -100)
    arr = make_img(mapinfo, map_name, list(lmks_around) + [landmark],
                   landmark_border=landmark_border,
                   landmark_values=lmk_vals)
    return arr


def make_nbr_img(mapinfo, map_name,
                 landmark, landmark_border=True,
                 dist_factor=2.0):
    """Given single landmark, plot the neigbor building and streets."""
    dims = mapinfo.xyrange(landmark, map_name)
    dist = max(dims) * dist_factor
    lmk_ctr = mapinfo.center_of_mass(landmark, map_name)
    lmks_around = set()
    bdgs_around = set()
    lmk_vals = {}
    for cell in sorted(mapinfo.cell_to_landmark(map_name)):
        if euclidean_dist(lmk_ctr, cell) < dist:
            lmk = mapinfo.landmark_at(map_name, cell)
            if lmk == landmark:
                lmk_vals[lmk] = (100, -100)
            else:
                lmks_around.add(lmk)
                if lmk in mapinfo.streets(map_name):
                    # Color streets differently (unless they are the landmark itself)
                    lmk_vals[lmk] = (300, -300)
                else:
                    lmk_vals[lmk] = (200, -200)
    arr = make_img(mapinfo, map_name, list(lmks_around) + [landmark],
                   landmark_border=landmark_border,
                   landmark_values=lmk_vals)
    return arr

def to_ego_img(arr, map_name, landmark, mapinfo):
    """Shifts the center of the image to be the center of mass of the landmark"""
    cur_ctr = (arr.shape[0] // 2, arr.shape[1] // 2)
    lm_ctr = mapinfo.center_of_mass(landmark, map_name)
    dx = cur_ctr[0] - lm_ctr[0]
    dy = cur_ctr[1] - lm_ctr[1]
    new_arr = np.zeros(arr.shape)
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            try:
                new_arr[x+dx, y+dy] = arr[x,y]
            except IndexError:
                pass
    return new_arr

def ego_lmk_map(landmark, map_name, mapinfo,
                mapsize=(28,28), landmark_border=True, use_context=True):
    if use_context:
        arr = make_nbr_img(mapinfo, map_name, landmark, landmark_border=landmark_border)
    else:
        arr = make_img(mapinfo, map_name, [landmark], landmark_border=landmark_border)
    arr = to_ego_img(arr, map_name, landmark, mapinfo)
    w,l = mapinfo.map_dims(map_name)
    tl = ((w-mapsize[0])//2, (l-mapsize[1])//2)
    return arr[tl[0]:tl[0]+mapsize[0],
               tl[1]:tl[1]+mapsize[1]]

def rotate_img(arr, angle):
    """Rotate by angle counter clock wise"""
    arr = ndimage.rotate(arr, math.degrees(angle), reshape=False, order=0)
    return arr

def rotate_pt(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def translate_img(arr, vec):
    new_arr = np.zeros(arr.shape)
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            try:
                new_arr[x+vec[0], y+vec[1]] = arr[x,y]
            except IndexError:
                pass
    return new_arr

def translate_point(point, vec):
    return np.asarray(point) + np.asarray(vec)


def scale_map(arr, dims):
    """
    Given an image array (arr), and a `dim`, tuple (width, length),
    """
    new_arr = np.zeros(dims)
    if dims[0] > arr.shape[0]\
       and dims[1] > arr.shape[1]:
        # Going from low-res to high-res
        for x in range(dims[0]):
            for y in range(dims[1]):
                arr_x, arr_y = scale_point((x,y), dims, arr.shape)
                new_arr[x,y] = arr[arr_x, arr_y]
    elif dims[0] < arr.shape[0]\
       and dims[1] < arr.shape[1]:
        # Going from high-res to low res
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                new_x, new_y = scale_point((x,y), arr.shape, dims)
                new_arr[new_x, new_y] += arr[x,y]
        new_arr[new_arr >= 0.5] = 1.0
        new_arr[new_arr < 0.5] = 0.0
    elif dims[0] == arr.shape[0]\
       and dims[1] == arr.shape[1]:
        new_arr = arr
    else:
        raise ValueError("Either upscale or downscale on both axes together.")
    return new_arr

def scale_point(point, old_dims, new_dims):
    x, y = point
    old_w, old_l = old_dims
    new_w, new_l = new_dims
    new_x = max(0, min(new_w-1, int(round(x*new_w / old_w))))
    new_y = max(0, min(new_l-1, int(round(y*new_l / old_l))))
    return new_x, new_y

def zoom_map(arr, scale=2.0):
    """Zoom the given image (2d array). Return an array
    that is of the same shape as arr, except the content
    of the array appears to be zoomed in. The zoom happens
    with respect to the center of the image."""
    new_arr = ndimage.zoom(arr, scale, order=0, mode="constant")
    # If the new array become smaller, place it at the center of
    # an array with the same size as `arr`
    if scale < 1.0:
        container = np.zeros(arr.shape)
        shiftx = (container.shape[0] - new_arr.shape[0]) // 2
        shifty = (container.shape[1] - new_arr.shape[1]) // 2
        container[shiftx:shiftx + new_arr.shape[0],
                  shifty:shifty + new_arr.shape[1]] = new_arr
        new_arr = container
    return new_arr

def zoom_point(point, dims, scale=2.0, bound=True):
    """
    Zoom the point; Compute a new location of the point
    that will be `scale` times the distance from the center
    of the rectangle specified by `dimes`, than the original point.

    if `bound` is True, then if the zoomed point is out of bound,
    we'll return None. Otherwise, just return the zoomed point always.
    """
    arr = np.zeros(dims)
    arr[point[0], point[1]] = 1
    new_arr = ndimage.zoom(arr, scale, order=0, mode="constant")
    new_point = np.argwhere(new_arr == 1)
    if len(new_point) == 0:
        # This zoom did not work. Probably because the scale is too low.
        return None

    new_point = np.round(np.mean(new_point, axis=0)).astype(int)  # Take the mean
    if scale < 1.0:
        container = np.zeros(dims)
        shiftx = (container.shape[0] - new_arr.shape[0]) // 2
        shifty = (container.shape[1] - new_arr.shape[1]) // 2
        new_point[0] += shiftx
        new_point[1] += shifty
    return tuple(new_point)


def to_polar(point):
    rho = np.linalg.norm(point)
    phi = np.arctan2(point[1], point[0])
    return np.array([rho, phi])

def to_cartesian(point):
    rho, phi = point
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))
