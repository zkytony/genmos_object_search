# Code adapted from https://github.com/zkytony/graphspn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines
import matplotlib.patheffects as path_effects

import os
import math
import random
import numpy as np

##########Topo-map related##########
def compute_view_number(node, neighbor, divisions=8, phase_shift=0):
    """
    Assume node and neighbor have the 'pose' attribute. Return an integer
    within [0, divisions-1] indicating the view number of node that is
    connected to neighbor.

    phase_shift (float): A permanent rotation in the views
        around the center (in rad).
    """
    x, y = node.coords[0], node.coords[1]
    nx, ny = neighbor.coords[0], neighbor.coords[1]
    angle_rad = math.atan2(ny-y, nx-x) - phase_shift
    if angle_rad < 0:
        angle_rad = math.pi*2 - abs(angle_rad)
    view_number = int(math.floor(angle_rad / (2*math.pi / divisions)))  # floor division
    return view_number


def compute_view_node_coords(center_coords, viewnum, dist,
                             divisions=8, phase_shift=0):
    """
    phase_shift (float): A permanent rotation in the views
        around the center (in rad).
    """
    angle_rad = viewnum * (2*math.pi / divisions) + phase_shift
    dx = dist * math.cos(angle_rad)
    dy = dist * math.sin(angle_rad)
    x, y = center_coords[0], center_coords[1]
    return x + dx, y + dy


def abs_view_distance(v1, v2, num_divisions=8):
    return min(abs(v1-v2), num_divisions-abs(v1-v2))


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


def compute_edge_pair_view_distance(edge1, edge2, dist_func=abs_view_distance, meeting_node=None):
    """
    Given two edges, first check if the two share one same node. If not,
    raise an error. If so, compute the absolute view distance between
    the two edges. The caller can optionally supply meeting_node, if
    already known.
    """
    if meeting_node is None:
        for n1 in edge1.nodes:
            for n2 in edge2.nodes:
                if n1 == n2:
                    meeting_node = n1
    if meeting_node is None:
        raise ValueError("edge %s and edge %s do not intersect!" % (edge1, edge2))
    other_nodes = []
    for edge in (edge1, edge2):
        i = edge.nodes.index(meeting_node)
        other_nodes.append(edge.nodes[1-i])

    # Compute view numbers and distance
    v1 = compute_view_number(meeting_node, other_nodes[0])
    v2 = compute_view_number(meeting_node, other_nodes[1])
    return dist_func(v1, v2)


################# Plotting ##################
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


def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))


################# Colors ##################
def linear_color_gradient(rgb_start, rgb_end, n):
    colors = [rgb_start]
    for t in range(1, n):
        colors.append(tuple(
            rgb_start[i] + float(t)/(n-1)*(rgb_end[i] - rgb_start[i])
            for i in range(3)
        ))
    return colors

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
