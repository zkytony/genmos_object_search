import utils
import json
import parse_osm_mapping
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
import pickle
import math

class GridCell(object):
    def __init__(self, nw, ne, sw, se):
        """
        ne: tuple representing lat/lon of northeast corner of grid cell
        nw: tuple representing lat/lon of northwest corner of grid cell
        sw: tuple representing lat/lon of southwest corner of grid cell
        se: tuple representing lat/lon of southeast corner of grid cell
        """
        # four corners
        self.nw = nw
        self.ne = ne
        self.sw = sw
        self.se = se

    def toJSON(self):
        return json.dumps(self, default=lambda o: vars(o), sort_keys=True, indent=4)

    # Shapely polygon representation
    def get_poly(self):
        return Polygon([self.nw, self.sw, self.se, self.ne])

        # an approximate center of the cell
        # self.center = ((ne[0] + se[0]) / 2.0, (ne[1] + nw[1]) / 2.0)

    def is_in(self, lat, lon):
        """
        Determine if a lat/lon point is within this cell.
        """
        return lat >= self.se[0] and lat <= self.ne[0] \
            and lon >= self.nw[1] and lon <= self.ne[1]

#     def __dict__(self):
#         return {"nw": self.nw, "ne": self.ne, "sw": self.sw, "se": self.se}
#
# def jdefault(obj):
#     return obj.__dict__

def create_map(name, coords):
    CELL_DIAM = 2
    MAP_DIAM = 50 # 50m padding

    idx = 0

    idx_to_cell = {}
    idx_to_latlon = {}
    idx_to_content = {}
    idx_to_occupancy = {}

    # Brown campus
    # the point is a center point
    nw_limit, ne_limit, sw_limit, se_limit = utils.bbox(coords, MAP_DIAM)
    map_data = utils.create_map_from_osm(coords, MAP_DIAM)
    map_data = parse_osm_mapping.parse_json(map_data)
    parse_osm_mapping.build_ways(map_data)
    ways, ways_content = parse_osm_mapping.get_ways()
    nodes, nodes_content = parse_osm_mapping.get_nodes()
    occupied_set = parse_osm_mapping.get_occupied()

    north_limit = nw_limit[0]
    east_limit = se_limit[1]
    south_limit = se_limit[0]
    west_limit = nw_limit[1]

    curr_nw = nw_limit
    curr_ne = ne_limit
    curr_sw = sw_limit
    curr_se = se_limit

    # build the idx_to_cell dictionary
    # set indices from west to east, north to south

    side_length = (MAP_DIAM * 2) / CELL_DIAM
    side_length = math.ceil(side_length)
    for i in range(0, side_length):
        for j in range(0, side_length):
            nw, ne, sw, se = utils.bbox_cell(curr_nw, CELL_DIAM)
            cell = GridCell(nw, ne, sw, se)
            inner_dict = {"nw": nw, "ne": ne, "sw": sw, "se":se}
            idx_to_latlon[idx] = inner_dict
            idx_to_cell[idx] = cell

            # go to next cell in map
            idx += 1
            curr_nw = ne
            curr_sw = se
        curr_nw = (curr_sw[0], west_limit)

    for way_key in ways:
        way_length = len(ways[way_key])
        if (way_length > 2):
            ways[way_key] = Polygon(ways[way_key])
            way_poly = ways[way_key]
            for idx in idx_to_cell:
                cell_poly = idx_to_cell[idx].get_poly()
                if cell_poly.intersects(way_poly):
                    idx_to_content[idx] = ways_content[way_key]
                    if way_key in occupied_set:
                        idx_to_occupancy[idx] = 1
                    else:
                        idx_to_occupancy[idx] = 0
        elif (way_length == 2):
            ways[way_key] = LineString(ways[way_key])
            way_poly = ways[way_key]
            for idx in idx_to_cell:
                cell_poly = idx_to_cell[idx].get_poly()
                if cell_poly.intersects(way_poly):
                    idx_to_content[idx] = ways_content[way_key]
                    if way_key in occupied_set:
                        idx_to_occupancy[idx] = 1
                    else:
                        idx_to_occupancy[idx] = 0
        elif (way_length == 1):
            ways[way_key] == Point(ways[way_key][0], ways[way_key][1])
            way_poly = ways[way_key]
            for idx in idx_to_cell:
                cell_poly = idx_to_cell[idx].get_poly()
                if cell_poly.contains(way_poly):
                    idx_to_content[idx] = ways_content[way_key]
                    if way_key in occupied_set:
                        idx_to_occupancy[idx] = 1
                    else:
                        idx_to_occupancy[idx] = 0

    for node_key in nodes:
        nodes[node_key] = Point(nodes[node_key][0], nodes[node_key][1])
        node_pt = nodes[node_key]
        for idx in idx_to_cell:
            cell_poly = idx_to_cell[idx].get_poly()
            if cell_poly.contains(node_pt):
                idx_to_content[idx] = nodes_content[node_key]
                if node_key in occupied_set:
                    idx_to_occupancy[idx] = 1
                else:
                    idx_to_occupancy[idx] = 0


    with open("idx_to_latlon" + name + ".json", "w") as fout:
        json.dump(idx_to_latlon, fout)

    with open("idx_to_content" + name + ".json", "w") as fout:
        json.dump(idx_to_content, fout)

    with open("idx_to_occupancy" + name + ".json", "w") as fout:
        json.dump(idx_to_occupancy, fout)

    ### ASSUMPTIONS ###
    # only one landmark in each cell

if __name__ == "__main__":
    cities = {"new_york1": (40.7164913, -73.9962504), \
    "new_york2": (40.7534810, -73.9808880), \
    "los_angeles1": (34.0590101, -118.4438383), \
    "los_angeles2": (34.0617454, -118.3003711), \
    "chicago1": (41.9973874, -87.7637780), \
    "chicago2": (41.9438833, -87.6492669), \
    "houston1": (29.7152679, -95.4381848), \
    "houston2": (29.6896316, -95.4084937), \
    "phoenix1": (33.4950345, -111.9261851), \
    "phoenix2": (33.5056714, -112.0444816), \
    "philadelphia1": (39.9521727, -75.1452280), \
    "philadelphia2": (39.9494622, -75.1975287), \
    "san_diego1": (32.7114860, -117.1601106), \
    "san_diego2": (32.7219451, -117.1675129), \
    "jacksonville1": (30.2888478, -81.3948911), \
    "jacksonville2": (30.2902513, -81.3898624), \
    "columbus1": (39.9602716, -82.9884074), \
    "columbus2": (39.9703422, -83.0074057), \
    "charlotte1": (35.2171966, -80.8304550), \
    "charlotte2": (35.2205909, -80.8129973), \
    "indianapolis1": (39.7527812, -86.1403024), \
    "indianapolis2": (39.7684285, -86.1578958), \
    "seattle1": (47.5975203, -122.3250451), \
    "seattle2": (47.6102141, -122.3367371), \
    "denver1": (39.7369283, -104.9896741), \
    "denver2": (39.6801057, -104.9407819), \
    "washington_dc1": (38.9335071, -77.0571410), \
    "washington_dc2": (38.8764024, -77.0121885), \
    "boston1": (42.3500897, -71.0773484), \
    "boston2": (42.3510563, -71.064923)}

    for name, coords in cities.items():
        create_map(name, coords)
