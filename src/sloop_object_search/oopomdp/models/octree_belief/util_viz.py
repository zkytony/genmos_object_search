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

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcl
import matplotlib.lines as lines
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pylab import rcParams
import numpy as np
rcParams['figure.figsize'] = 4,4

CMAPS = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
]

def plot_voxels(vp, vr=None, vc=None, ax=None,
                edgecolor=None, alpha=None, center=False, linewidth=1):
    """
    Given vp (voxel poses), vr (voxel resolutions), vl (voxel labels)
    cm (color mapping, from label to color), plot these voxels.

    If `center` is True, then the voxel poses are center of the voxels.
    If false, then the voxel pose is at a lower corner of the voxel.

    plot the voxels. Adapted from the answer here:
    https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
    """
    def cuboid_data(o, res=1, center=True):
        # I want the pose to be the center of the cube
        if center:
            X = [[[-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5]],
                 [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5]],
                 [[0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]],
                 [[-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]],
                 [[-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]],
                 [[-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5]]]
        else:
            X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
                 [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
                 [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                 [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
                 [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
                 [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:,:,i] *= res
        X += np.array(o)
        return X

    def plotCubeAt(positions, sizes=None, colors=None, center=True,
                   edgecolor=None, alpha=None, linewidth=1):
        if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
        if not isinstance(sizes,(list,np.ndarray)): sizes=[1]*len(positions)
        g = []
        for p,s,c in zip(positions,sizes,colors):
            g.append( cuboid_data(p, res=s, center=center) )
        pc = Poly3DCollection(np.concatenate(g), linewidth=linewidth)
        if isinstance(alpha, float):
            pc.set_alpha(alpha)
        pc.set_edgecolor(edgecolor)
        pc.set_facecolor(np.repeat(colors,6))
        return pc

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(1,1,1,projection="3d")
    # ax.set_aspect('equal')
    pc = plotCubeAt(vp, sizes=vr, colors=vc, center=center,
                    edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    ax.add_collection3d(pc)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(30, 50)
    return ax


def plot_points(xvals, yvals, color=None,
                size=1.5, label=None, connected=True, style="--", linewidth=1.5,
                xlabel='x', ylabel='f(x)', loc="lower right"):
    if not connected:
        plt.scatter(xvals, yvals, s=size, c=color, label=label)
    else:
        plt.plot(xvals, yvals, style, linewidth=linewidth, label=label)
    # plt.axhline(y=0, color='k')
    # plt.axvline(x=0, color='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)

def save_plot(path, bbox_inches='tight'):
    plt.savefig(path, bbox_inches=bbox_inches)
    plt.close()


# colors
def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent

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

def random_unique_color(colors, ctype=1):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % random.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % random.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % random.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % random.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % random.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = random.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = random.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color

# Plotting in matplotlib
def plot_dot(ax, px, py, color='blue', dotsize=2, fill=True, zorder=0, linewidth=0, edgecolor=None, label_text=None, alpha=1.0):
    very_center = plt.Circle((px, py), dotsize, facecolor=color, fill=fill, zorder=zorder, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha)
    ax.add_artist(very_center)
    if label_text:
        text = ax.text(px, py, label_text, color='white',
                        ha='center', va='center', size=7, weight='bold')
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])

        # t = ax.text(px-5, py-5, label_text, fontdict=font)
        # t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))
    return px, py


def plot_line(ax, p1, p2, linewidth=1, color='black', zorder=0, alpha=1.0):
    p1x, p1y = p1
    p2x, p2y = p2
    ax = plt.gca()
    line = lines.Line2D([p1x, p2x], [p1y, p2y], linewidth=linewidth, color=color, zorder=zorder,
                        alpha=alpha)
    ax.add_line(line)

### Plotting specifically for gmapping related stuffs.
def transform_coordinates(gx, gy, map_spec, img):
    # Given point (gx, gy) in the gmapping coordinate system (in meters), convert it
    # to a point or pixel in Cairo context. Cairo coordinates origin is at top-left, while
    # gmapping has coordinates origin at lower-left.
    imgHeight, imgWidth = img.shape
    res = float(map_spec['resolution'])
    originX = float(map_spec['origin'][0])  # gmapping map origin
    originY = float(map_spec['origin'][1])
    # Transform from gmapping coordinates to pixel cooridnates.
    return ((gx - originX) / res, imgHeight - (gy - originY) / res)

def plot_dot_map(ax, rx, ry, map_spec, img, color='blue', dotsize=2, fill=True, zorder=0, linewidth=0, edgecolor=None, label_text=None):
    px, py = transform_coordinates(rx, ry, map_spec, img)
    return plot_dot(ax, px, py, color=color,
                    dotsize=dotsize, fill=fill, zorder=zorder, linewidth=linewidth,
                    edgecolor=edgecolor, label_text=label_text)

def plot_line_map(ax, g1, g2, map_spec, img, linewidth=1, color='black', zorder=0, alpha=1.0):
    # g1, g2 are two points with gmapping coordinates
    p1x, p1y = transform_coordinates(g1[0], g1[1], map_spec, img)
    p2x, p2y = transform_coordinates(g2[0], g2[1], map_spec, img)
    plot_line(ax, (p1x, p1y), (p2x, p2y), linewidth=linewidth, color=color, zorder=zorder, alpha=alpha)

def zoom_plot(p, img, ax, zoom_level=0.35):
    # Zoom by setting limits. Center around p
    px, py = p
    h, w = img.shape
    sidelen = min(w*zoom_level*0.2, h*zoom_level*0.2)
    ax.set_xlim(px - sidelen/2, px + sidelen/2)
    ax.set_ylim(py - sidelen/2, py + sidelen/2)

def zoom_rect(p, img, ax, h_zoom_level=0.35, v_zoom_level=0.35):
    # Zoom by setting limits
    px, py = p
    h, w = img.shape
    xsidelen = w*h_zoom_level*0.2
    ysidelen = h*v_zoom_level*0.2
    ax.set_xlim(px - xsidelen/2, px + xsidelen/2)
    ax.set_ylim(py - ysidelen/2, py + ysidelen/2)
