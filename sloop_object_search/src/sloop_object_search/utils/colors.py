import random
import numpy as np
from .math import remap

# colors
def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)
    If `change_alpha` is True, then the alpha will also be redueced
    by the specified amount.'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent

def lighter_with_alpha(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)
    If `change_alpha` is True, then the alpha will also be redueced
    by the specified amount.'''
    color = np.array(color)
    white = np.array([255, 255, 255, 255])
    vector = white-color

    cc = color + vector*percent
    cc[3] = color[3] + (color-white)[3]*(percent)
    return cc

def linear_color_gradient(rgb_start, rgb_end, n):
    colors = [rgb_start]
    for t in range(1, n):
        colors.append(tuple(
            rgb_start[i] + float(t)/(n-1)*(rgb_end[i] - rgb_start[i])
            for i in range(3)
        ))
    return colors

def color_map(val, val_posts, rgb_posts):
    """Given a value that lies between an interval in valposts
    which should be [minval, post1, post2, ..., postn, maxval],
    paired with rgb posts, extrapolate the color for val.

    val_posts can be [minval, maxval] only. In that case,
    it will be expanded to contain intermediate values.

    val_posts should be a list of numbers or floats.
    rgb_posts should be a list of [r,g,b] arrays or tuples

    we have some predefined rgb_posts, such as COLOR_MAP_JET"""
    if len(val_posts) == 2:
        val_posts = np.linspace(val_posts[0], val_posts[1], num=len(rgb_posts))
    else:
        if len(val_posts) != len(rgb_posts):
            raise ValueError("number of values posts should equal"\
                             "the number of rgb posts. Or, specify [minval, maxval]")

    # determine which post val falls into
    for i in range(1, len(val_posts)):
        in_range = val_posts[i-1] <= val <= val_posts[i]
        if in_range:
            rgb_min = rgb_posts[i-1]
            rgb_max = rgb_posts[i]
            r = remap(val, val_posts[i-1], val_posts[i], rgb_min[0], rgb_max[0])
            g = remap(val, val_posts[i-1], val_posts[i], rgb_min[1], rgb_max[1])
            b = remap(val, val_posts[i-1], val_posts[i], rgb_min[2], rgb_max[2])
            return np.array([r, g, b])
    raise ValueError(f"value {val} does not fall in any interval in {val_posts}")

class cmaps:
    COLOR_MAP_JET = [[0.07, 0.0, 0.43],
                     [0.93, 0.53, 0.0],
                     [0.0, 0.9, 0.93],
                     [0.43, 0.07, 0.0]]

    COLOR_MAP_GRAYS = [[0.95, 0.95, 0.95],
                       [0.05, 0.05, 0.05]]

    # This one goes into darker gray quicker
    COLOR_MAP_GRAYS2 = [[0.95, 0.95, 0.95],
                        [0.5, 0.5, 0.5],
                        [0.2, 0.2, 0.2],
                        [0.05, 0.05, 0.05]]

    COLOR_MAP_HALLOWEEN = [[0.98, 0.82, 0.55],
                           [0.53, 0.52, 0.48],
                           [0.05, 0.05, 0.05]]


def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

def inverse_color_rgb(rgb):
    r,g,b = rgb
    return (255-r, 255-g, 255-b)

def inverse_color_hex(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    return inverse_color_rgb(hex_to_rgb(hx))

def random_unique_color(colors, ctype=1, rnd=random, fmt="rgb"):
    colors_hex = []
    for c in colors:
        if not c.startswith("#"):
            colors_hex.append(rgb_to_hex(c))
        else:
            colors_hex.append(c)
    color = _random_unique_color_hex(colors, ctype=ctype, rnd=rnd)
    if fmt == "rgb":
        return hex_to_rgb(color)
    else:
        return color

def _random_unique_color_hex(colors, ctype=1, rnd=random):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % rnd.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % rnd.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % rnd.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % rnd.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % rnd.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % rnd.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % rnd.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % rnd.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = rnd.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = rnd.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color

def mean_rgb(rgb_array):
    """
    rgb_array is a numpy array of shape (w, l, 3)
    """
    return np.mean(rgb_array.reshape(-1, 3), axis=0).astype(rgb_array.dtype)


__all__ = ['lighter',
           'lighter_with_alpha',
           'rgb_to_hex',
           'hex_to_rgb',
           'inverse_color_rgb',
           'inverse_color_hex',
           'random_unique_color',
           'mean_rgb']
