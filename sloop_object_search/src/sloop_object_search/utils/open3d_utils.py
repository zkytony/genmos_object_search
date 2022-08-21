import open3d as o3d

def cube_unfilled(scale=1, color=[1,0,0]):
    # http://www.open3d.org/docs/0.9.0/tutorial/Basic/visualization.html
    if hasattr(scale, "__len__"):
        scale_x, scale_y, scale_z = scale
    else:
        scale_x = scale_y = scale_z = scale

    points = [
        [0,        0,       0],
        [scale_x,  0,       0],
        [0,        scale_y, 0],
        [scale_x,  scale_y, 0],
        [0,        0,       scale_z],
        [scale_x,  0,       scale_z],
        [0,        scale_y, scale_z],
        [scale_x,  scale_y, scale_z],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set
