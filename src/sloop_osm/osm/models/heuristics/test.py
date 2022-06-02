# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from sloop.models.heuristics.rules import *
from sloop.models.nn.plotting import plot_foref
import numpy as np
import math


building1=\
"""
...........
..x..x.....
..xxxxx....
..x..x.....
..xxxxx....
..x..x.....
..xxxxx....
...........
"""


building2=\
"""
...xxx...
..xx.......
..xxx......
..xx.......
"""

building3=\
"""
...................
.............xxxxxx
........xxxxx......
....xxxx...........
.xxx...............
xx.................
"""

building4=\
"""
...................
.............xxxxxx
xxxxxxxxxxxxxxxx...
...................
"""

building5=\
"""
.....
..x..
.....
"""

building6=\
"""
..........
..........
.....xx...
....xx....
..........
..........
"""

def place_landmark(map_arr, landmark_str, origin):
    lines = landmark_str.split("\n")
    landmark_width = len(lines[0])
    landmark_length = len(lines)

    ox, oy = origin
    if len(np.unique(map_arr[oy:oy+landmark_length,
                             ox:ox+landmark_length])) != 1:
        raise ValueError("Cannot place landmark here %s" % (origin))

    landmark_footprint = set({})  # set of indices the landmark actually cover.
    for dy in range(len(lines)):
        for dx in range(len(lines[dy])):
            c = lines[dy][dx]
            if c == ".":
                map_arr[oy+dy, ox+dx] = 0
            elif c == "x":
                map_arr[oy+dy, ox+dx] = 1
                landmark_footprint.add((ox+dx, oy+dy))

    return landmark_footprint


def plot_map(map_arr, ax=None):
    xvals = []
    yvals = []
    for y, x in np.argwhere(map_arr == 1):
        xvals.append(x)
        yvals.append(y)
    if ax is None:
        ax = plt.gca()
    ax.scatter(xvals, yvals, zorder=2, color="orange")


def plot_rule(rule, map_dims, footprint, ax=None, **kwargs):
    belief = {}
    belief = rule.compute(list(footprint), map_dims, **kwargs)
    plot_belief(belief, ax=ax)


def plot_multi(rules, map_dims, footprints, ax=None, kwargs={}):
    belief = None
    for i in range(len(rules)):
        if rules[i] in kwargs:
            b = rules[i].compute(list(footprints[i]), map_dims, **kwargs[rules[i]])
        else:
            b = rules[i].compute(list(footprints[i]), map_dims)

        if belief is None:
            belief = b
        else:
            belief = combine_beliefs(belief, b)
    plot_belief(belief, ax=ax)


def plot_belief(belief, ax=None):
    xvals, yvals, c = [], [], []
    for x, y in belief:
        xvals.append(x)
        yvals.append(y)
        c.append(belief[(x,y)])
    if ax is None:
        ax = plt.gca()
    ax.scatter(xvals, yvals, c=c, marker='s', alpha=0.6)


def test():
    plt.figure(figsize=(4,4), facecolor='w', edgecolor='k')
    map_arr = np.zeros((40, 40))
    footprint = place_landmark(map_arr, building6, (13,27))
    plot_map(map_arr)

    behind = ForefRule("behind")
    center = np.mean(np.array(list(footprint)), axis=0)
    foref = [13., 27., -5.70686784]
    plot_foref(foref, plt.gca(), plot_perp=False)
    plot_multi([behind], map_arr.shape,
               [footprint], kwargs={behind: {"foref":foref}})

    plt.xlim(0, map_arr.shape[1])
    plt.ylim(0, map_arr.shape[0])
    plt.gca().invert_yaxis()
    plt.savefig("test_fig.png")


if __name__ == "__main__":
    test()

    # footprint1 = place_landmark(map_arr, building1, (20,20))
    # footprint2 = place_landmark(map_arr, building2, (25,10))
    # # footprint3 = place_landmark(map_arr, building3, (10,15))
    # # footprint4 = place_landmark(map_arr, building4, (20,30))

    # plot_map(map_arr)

    # near = NearRule()
    # beyond = BeyondRule()
    # against = AgainstRule()
    # at = AtRule()
    # east = DirectionRule("east")
    # west = DirectionRule("west")
    # south = DirectionRule("south")
    # north = DirectionRule("north")
    # southwest = DirectionRule("southwest")
    # northwest = DirectionRule("northwest")
    # northeast = DirectionRule("northeast")

    # front = ForefRule("front")
    # behind = ForefRule("behind")
    # left = ForefRule("left")
    # right = ForefRule("right")
    # center1 = np.mean(np.array(list(footprint1)), axis=0)
    # # center2 = np.mean(np.array(list(footprint2)), axis=0)
    # # center3 = np.mean(np.array(list(footprint3)), axis=0)
    # # center4 = np.mean(np.array(list(footprint4)), axis=0)
    # foref1 = [*center1, math.radians(90)]
    # # foref2 = [*center2, math.radians(0)]
    # # foref3 = [*center3, math.radians(-15)]
    # # foref4 = [*center4, math.radians(-180)]

    # plot_multi([against], map_arr.shape,
    #            [footprint1])


    # plot_foref(foref1, plt.gca())
    # # plot_foref(foref3, plt.gca())
    # # plot_foref(foref2, plt.gca())
    # # plot_foref(foref4, plt.gca())
    # # plot_foref(foref1, plt.gca())
    # # plot_multi([north, at], map_arr.shape, [footprint1, footprint2])
    # # plot_multi([east, beyond], map_arr.shape, [footprint1, footprint2])
