import utils
import json
import parse_osm_mapping
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

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

    """
    def __dict__(self):
        return {"nw": self.nw, "ne": self.ne, "sw": self.sw, "se": self.se}

    def __str__(self):
        return "{nw: " + str(self.nw) + ", ne: " + str(self.ne) + ", sw: ", + str(self.sw) + ", se: ", + str(self.se) + "}"
    """

def jdefault(obj):
    return obj.__dict__

CELL_DIAM = 50
MAP_DIAM = 500

idx_to_cell = {}
name_to_idx = {}
idx = 0

# Brown campus
# the point is a center point
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.826771, -71.40255), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.826771, -71.40255), MAP_DIAM)

# NYC
nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((40.7736696, -73.9596624), MAP_DIAM)
map_data = utils.create_map_from_osm((40.7736696, -73.9596624), MAP_DIAM)

map_data = parse_osm_mapping.parse_json(map_data)
parse_osm_mapping.build_ways(map_data)
ways = parse_osm_mapping.get_ways()
nodes = parse_osm_mapping.get_nodes()
buildings = parse_osm_mapping.get_buildings()
building_polys = []

north_limit = nw_limit[0]
east_limit = se_limit[1]
south_limit = se_limit[0]
west_limit = nw_limit[1]

print("north limit: ", north_limit)
print("east limit: ", east_limit)
print("south limit: ", south_limit)
print("west limit: ", west_limit)

curr_nw = nw_limit
curr_ne = ne_limit
curr_sw = sw_limit
curr_se = se_limit

# build the idx_to_cell dictionary
# set indices from west to east, north to south
while curr_nw[0] > south_limit:
    while curr_nw[1] < east_limit:
        nw, ne, sw, se = utils.bbox_cell(curr_nw, CELL_DIAM)
        cell = GridCell(nw, ne, sw, se)
        idx_to_cell[idx] = cell
        idx += 1
        curr_nw = ne
        curr_sw = se
    curr_nw = (curr_sw[0], west_limit)

for building in buildings:
    building_polys.append(Polygon(building))

for way_key in ways:
    way_length = len(ways[way_key])
    if (way_length > 2):
        ways[way_key] = Polygon(ways[way_key])
        way_poly = ways[way_key]
        for idx in idx_to_cell:
            cell_poly = idx_to_cell[idx].get_poly()
            if cell_poly.intersects(way_poly):
                if name_to_idx.get(way_key) == None:
                    name_to_idx[way_key] = [idx]
                else:
                    name_to_idx[way_key].append(idx)
    elif (way_length == 2):
        ways[way_key] = LineString(ways[way_key])
        way_poly = ways[way_key]
        for idx in idx_to_cell:
            cell_poly = idx_to_cell[idx].get_poly()
            if cell_poly.intersects(way_poly):
                if name_to_idx.get(way_key) == None:
                    name_to_idx[way_key] = [idx]
                else:
                    name_to_idx[way_key].append(idx)
    elif (way_length == 1):
        ways[way_key] == Point(ways[way_key][0], ways[way_key][1])
        way_poly = ways[way_key]
        for idx in idx_to_cell:
            cell_poly = idx_to_cell[idx].get_poly()
            if cell_poly.contains(way_poly):
                if name_to_idx.get(way_key) == None:
                    name_to_idx[way_key] = [idx]
                else:
                    name_to_idx[way_key].append(idx)

for node_key in nodes:
    nodes[node_key] = Point(nodes[node_key])
    landmark = nodes[node_key]


    # Check if a named node is in a building
    for building in building_polys:
        if building.intersects(nodes[node_key]):
            # treat the node like the building instead
            landmark = building

    for idx in idx_to_cell:
        cell_poly = idx_to_cell[idx].get_poly()
        if cell_poly.intersects(landmark):
            if name_to_idx.get(node_key) == None:
                name_to_idx[node_key] = [idx]
            else:
                name_to_idx[node_key].append(idx)


idx_to_cell_json = json.dumps(idx_to_cell, default=jdefault)
name_to_idx_json = json.dumps(name_to_idx, default=jdefault)

idx_to_cell_file = open("idx_to_cell_nyc.json", "w")
idx_to_cell_file.write(idx_to_cell_json)
idx_to_cell_file.close()

name_to_idx_file = open("name_to_idx_nyc.json", "w")
name_to_idx_file.write(str(name_to_idx_json))
name_to_idx_file.close()
