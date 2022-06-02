import numpy as np

######## Map utlities
def point_to_cell(map_to_point, idx_to_cell):
    """
    Returns a dictionary of map names to cell indices that contain the points.

    Parameters:
    - point dict: Dictionary of map name to lat/lon coordinate
        {"28_8": (-41, 71)}
    - idx to cell: Dictionary of grid cell index to cell bounding coordinates
        {"0": {"sw": (-41, 71), "ne": ...}}
    """
    map_to_idx = {}
    for map, point in map_to_point.items():
        # print(map)
        for idx, cell in idx_to_cell.items():
            # check that point is within the cell
            if point[1] >= cell["sw"][1] and point[1] <= cell["se"][1] \
               and point[0] >= cell["sw"][0] and point[0] <= cell["nw"][0]:
                map_to_idx[map] = int(idx)
    return map_to_idx

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def convert2pol(obj, lm):
    obj_rho, obj_phi = cart2pol(obj[0], obj[1])
    lm_rho, lm_phi = cart2pol(lm[0], lm[1])
    input = [obj[0], obj[1], lm[0], lm[1], obj_rho, obj_phi, lm_rho, lm_phi]
    return input

def convert2pol_center(obj, lm):
    obj_x = obj[0] - lm[0]
    obj_y = obj[1] - lm[1]
    obj_rho, obj_phi = cart2pol(obj_x, obj_y)
    lm_rho, lm_phi = cart2pol(lm[0], lm[1])
    input = [obj_x, obj_y, obj_rho, obj_phi]
    return input

def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

def pixel2grid(pixel, img_width, img_length, map_info, city_name, pixel_origin=(0,0)):
    """
    Converts human pixel annotation to pomdp coords.
    Needs to go through lattitude/longitude due to earth curvature
    """
    if city_name not in map_info.cardinal_to_limit:
        raise ValueError("Did you load %s into the map info?" % city_name)
    # City map corner limits
    e_lim = map_info.cardinal_to_limit[city_name]["E"]
    w_lim = map_info.cardinal_to_limit[city_name]["W"]
    n_lim = map_info.cardinal_to_limit[city_name]["N"]
    s_lim = map_info.cardinal_to_limit[city_name]["S"]

    # Interpolate lat/lon for the given pixel
    px, py = pixel
    ox, oy = pixel_origin
    lon = remap(px, ox, ox + img_width, w_lim, e_lim)  # Lattitude is a horizontal curve
    lat = remap(py, oy, oy + img_length, n_lim, s_lim)  # Longitude is a vertical curve

    # Find grid cell index and return the pomdp coordinates
    for idx in map_info.idx_to_cell[city_name]:
        south, west, north, east = map_info.cell_limits_latlon(city_name, idx)
        if west <= lon <= east\
           and south <= lat <= north:
            # return pomdp coord
            return map_info.idx_to_pomdp(city_name, int(idx))

def grid2pixels(grid_coord, img_width, img_length, map_info,
        city_name, pixel_origin=(0,0)):
    """
    Converts a pomdp grid into pixels at corners of the pomdp grid.
    Needs to go through lattitude/longitude due to earth curvature
    """
    # Obtain idx
    idx = map_info.pomdp_to_idx(city_name, grid_coord)
    if idx is None:
        return None

    # Obtain four corners
    cell_corners = map_info.idx_to_cell[city_name][str(idx)]  # TODO: consistent idx type!
    e_lim = map_info.cardinal_to_limit[city_name]["E"]
    w_lim = map_info.cardinal_to_limit[city_name]["W"]
    n_lim = map_info.cardinal_to_limit[city_name]["N"]
    s_lim = map_info.cardinal_to_limit[city_name]["S"]

    # Find the grid cell index
    pixels = []
    ox, oy = pixel_origin
    for direction, (lat, lon) in cell_corners.items():
        px = remap(lon, w_lim, e_lim, ox, ox + img_width)
        py = remap(lat, n_lim, s_lim, oy, oy + img_length)
        pixels.append((px, py))
    return pixels
