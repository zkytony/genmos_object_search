import numpy as np
import matplotlib.pyplot as plt
from sloop.datasets.dataloader import *

def plot_foref(foref, ax, c1="magenta", c2="lime",
               width=3.0, label=None, plot_perp=True, alpha=1.0,
               plot_arrow=True):
    """Plot the frame of reference `foref`.
    If plot_perp is True, then a second vector will be plotted with the color c2."""
    p1 = foref[:2]
    if len(foref) == 2:
        # It's just the origin.
        ax.scatter([p1[0]], [p1[1]], s=100, c="black")
        ax.scatter([p1[0]], [p1[1]], s=50, c=c1, alpha=0.8, label=label)
        return

    theta = foref[2]
    p2 = [p1[0] + math.cos(theta) * 6,
          p1[1] + math.sin(theta) * 6]

    p3 = [p1[0] + math.cos(theta + math.radians(90)) * 6,
          p1[1] + math.sin(theta + math.radians(90)) * 6]

    markers_on = [1]
    if plot_arrow:
        ax.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], head_width=width/5, head_length=width/5,
                 fc=c1, ec=c1, alpha=alpha, linewidth=width)
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "->", c=c1, linewidth=width, alpha=0.8,
                label=label, markevery=markers_on)
    if plot_perp:
        if plot_arrow:
            ax.arrow(p1[0], p1[1], p3[0]-p1[0], p3[1]-p1[1], head_width=width*1.5, head_length=width*1.5,
                     fc=c2, ec=c2, alpha=alpha, linewidth=width)
        else:
            ax.plot([p1[0], p3[0]], [p1[1], p3[1]], "->", c=c2, linewidth=width,
                    alpha=0.8, markevery=markers_on)
    ax.scatter([p1[0]], [p1[1]], s=100, c="black", marker="o")

def read_foref(dataset, foref_norm, relative=False):
    """Given a prediction of frame of reference that
    is normalized, recover the actual frame of reference unnoramlized."""
    foref = np.array(list(map(float, foref_norm)))
    if relative:
        foref[:2] = dataset.rescale(FdFoRefOriginRel.NAME, foref[:2])
    else:
        foref[:2] = dataset.rescale(FdFoRefOrigin.NAME, foref[:2])
    if len(foref_norm) > 2:
        foref[2] = dataset.rescale(FdFoRefAngle.NAME, foref[2])
    return foref

def plot_1d(values, title):
    minval = np.min(values)
    maxval = np.max(values)
    plt.hlines(0, minval - 0.1, maxval + 0.1)
    plt.xlim(minval - 0.1, maxval + 0.1)
    plt.ylim(-0.2, 0.2)
    y = np.zeros(np.shape(values))
    plt.plot(values, y, "|", linewidth=2, ms=40, alpha=0.25)
    plt.gca().set_yticks([])
    plt.title(title)


def plot_multiple(mapimg, forefs, objloc, colors, ax,
                  map_dims=(21,21), width_factor=1.0,
                  plot_perp=True, plot_obj=True,
                  plot_arrow=True,
                  leave_gap=False, alpha=0.8):
    """Asssume frame of references in `forefs` have
    already been rescaled.
    `leave_gap` is true if we want to have difference in
    the foref widths so that when they overlap you can see both."""
    im = ax.imshow(mapimg.transpose(), cmap="YlGnBu")
    ax.set_xlabel("x")
    ax.set_xlabel("y")
    ax.set_xlim(0, map_dims[0])
    ax.set_ylim(0, map_dims[1])
    ax.invert_yaxis()

    # Plot the object location
    if plot_obj:
        ax.scatter([objloc[0].item()], [objloc[1].item()], s=100, c="blue")

    width = len(forefs) * width_factor
    for label in forefs:
        foref = forefs[label]
        c1 = colors[label][0]
        c2 = colors[label][1]
        plot_foref(foref, ax, c1=c1, c2=c2,
                   width=width, label=label,
                   plot_perp=plot_perp,
                   plot_arrow=plot_arrow,
                   alpha=alpha)
        if leave_gap:
            width -= 1

    if not plot_arrow:
        # Get the legend
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_map(ax, m, alpha=1.0):
    ax.imshow(m, alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, m.shape[0])
    ax.set_ylim(0, m.shape[1])
    ax.invert_yaxis()
