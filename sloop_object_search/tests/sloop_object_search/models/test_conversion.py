import pomdp_py
import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sloop_object_search.utils.conversion import Frame, convert, convert_cov


def plot_gaussian(mean, cov, title):
    gaussian = pomdp_py.Gaussian(mean, cov)
    xs, ys = [], []
    for i in range(10000):
        p = gaussian.random()
        xs.append(p[0])
        ys.append(p[1])

    fig = plt.gcf()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xs, ys, alpha=0.2)
    ax.scatter([mean[0]], [mean[1]], c='black')
    ax.set_title(title)
    plt.show(block=False)
    plt.pause(1)
    ax.clear()
    fig.clear()


def test():
    np.random.seed(100)
    point_world = np.random.uniform(0, 10, (2,))
    region_origin = np.array([point_world[0] - 3,
                              point_world[1] - 2])
    search_space_resolution = 0.5
    cov_world = np.array([[2.0, 0.5],
                          [0.95, 1.5]])
    plot_gaussian(point_world.tolist(),
                  cov_world.tolist(), "World Space Gaussian")

    # Now, create a gaussian in pomdp space
    point_pomdp = convert(point_world, Frame.WORLD, Frame.POMDP_SPACE,
                          region_origin=region_origin,
                          search_space_resolution=search_space_resolution)
    cov_pomdp = convert_cov(cov_world, Frame.WORLD, Frame.POMDP_SPACE,
                          search_space_resolution=search_space_resolution)
    plot_gaussian(list(point_pomdp),
                  cov_pomdp.tolist(), "POMDP Space Gaussian")

    # Now, create a gaussian in world space from pomdp space
    point_world2 = convert(point_pomdp, Frame.POMDP_SPACE, Frame.WORLD,
                           region_origin=region_origin,
                           search_space_resolution=search_space_resolution)
    cov_world2 = convert_cov(cov_pomdp, Frame.POMDP_SPACE, Frame.WORLD,
                          search_space_resolution=search_space_resolution)
    plot_gaussian(list(point_world2),
                  cov_world2.tolist(), "World Space Gaussian (AGAIN)")

if __name__ == "__main__":
    test()
