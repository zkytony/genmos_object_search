import pytest
import open3d as o3d
from genmos_object_search.utils import open3d_utils


def test_draw_transparent_cube():
    g = open3d_utils.cube_filled(alpha=0.5)
    o3d.visualization.draw(g)


if __name__ == "__main__":
    test_draw_transparent_cube()
