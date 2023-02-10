import utils
import json
import parse_osm_mapping
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random
import numpy as np
from time import time
from shapely.geometry import Point, LineString, box
from shapely.geometry.polygon import Polygon
import sys
import os
from city_centers import CITY_CENTERS
from spatial_lang.data.hyperparams import CELL_DIAM, MAP_DIAM
import itertools
import copy

OSM_SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))

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

CITY_NAME = "houston"

# Brown campus
# the point is a center point

### Faunce House ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.826771, -71.40255), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.826771, -71.40255), MAP_DIAM)

### CVS on Thayer ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.8299353, -71.4007520), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.8299353, -71.4007520), MAP_DIAM)

### The Dorrance ##
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.8244409, -71.4123362), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.8244409, -71.4123362), MAP_DIAM)

### New York City ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((40.7736696, -73.9596624), MAP_DIAM)
# map_data = utils.create_map_from_osm((40.7736696, -73.9596624), MAP_DIAM)

### Los Angeles 1 ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((34.0590101, -118.4438383), MAP_DIAM)
# map_data = utils.create_map_from_osm((34.0590101, -118.4438383), MAP_DIAM)

### phoenix 2 ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((33.5056714, -112.0444816), MAP_DIAM)
# map_data = utils.create_map_from_osm((33.5056714, -112.0444816), MAP_DIAM)

### philadelphia2 ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((39.9494622, -75.1975287), MAP_DIAM)
# map_data = utils.create_map_from_osm((39.9494622, -75.1975287), MAP_DIAM)

### jacksonville new ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox(( 30.2889367, -81.3950641), MAP_DIAM)
# map_data = utils.create_map_from_osm(( 30.2889367, -81.3950641), MAP_DIAM)

### indianapolis2 ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((39.7684285, -86.1578958), MAP_DIAM)
# map_data = utils.create_map_from_osm((39.7684285, -86.1578958), MAP_DIAM)

### chicago ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((41.9973874, -87.7637780), MAP_DIAM)
# map_data = utils.create_map_from_osm((41.9973874, -87.7637780), MAP_DIAM)

### houston ###
# nw_limit, ne_limit, sw_limit, se_limit = utils.bbox((29.7563116, -95.3676771), MAP_DIAM)
# map_data = utils.create_map_from_osm((29.7563116, -95.3676771), MAP_DIAM)

# scraping osm
def scrape_osm(city_name):
    idx_to_cell = {}
    name_to_idx = {}
    idx = 0

    nw_limit, ne_limit, sw_limit, se_limit = utils.bbox(CITY_CENTERS[city_name], MAP_DIAM)
    map_data = utils.create_map_from_osm(CITY_CENTERS[city_name], MAP_DIAM)

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

    remove_keys = []
    for way_key in ways:
        way_length = len(ways[way_key])
        if (way_length > 2):
            ways[way_key] = Polygon(ways[way_key])
            way_poly = ways[way_key]
            print("way poly is valid: ", way_poly.is_valid)
            for idx in idx_to_cell:
                cell_poly = idx_to_cell[idx].get_poly()
                if cell_poly.intersects(way_poly):
                    if name_to_idx.get(way_key) == None:
                        name_to_idx[way_key] = [idx]
                    else:
                        name_to_idx[way_key].append(idx)
        else:
            remove_keys.append(way_key)

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

    for key in remove_keys:
        del ways[key]

    # minx, miny, maxx, maxy
    bbox = box(west_limit, south_limit, east_limit, north_limit)
    polyhole = Polygon(bbox.exterior.coords, [w.exterior.coords for w in ways.values()])
    return bbox, idx_to_cell, name_to_idx

# test_pt = Point(41.82388034255718, -71.41229219862235)
# test_pt = Point(41.82379675147419, -71.41240792417489)
# pixel_size = 455
# pixel_width = 698
# pixel_height = 696
# counter = 1

def generate_map_images(poly, num_maps, num_objs, city_name, idx_to_cell, name_to_idx):
    west, south, east, north = poly.bounds
    width = abs(east - west)
    height = abs(north - south)

    # annotations and offset comes from https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    # bike = OffsetImage(plt.imread("%s/object_images/red_bicycle_2.png" % OSM_SCRIPTS_PATH), zoom=0.01)
    # red_honda = OffsetImage(plt.imread("%s/object_images/red_honda.png" % OSM_SCRIPTS_PATH), zoom=0.05)
    # green_toyota = OffsetImage(plt.imread("%s/object_images/green_toyota.png" % OSM_SCRIPTS_PATH), zoom=0.05)
    #
    # objs = {"bike": bike, "rcar": red_honda, "gcar": green_toyota}
    objs = {"bike": ("%s/object_images/red_bicycle_2.PNG" % OSM_SCRIPTS_PATH, 0.01), "rcar": ("%s/object_images/red_honda.png" % OSM_SCRIPTS_PATH, 0.05), "gcar": ("%s/object_images/green_toyota.png" % OSM_SCRIPTS_PATH, 0.05)}
    obj_combos = list(itertools.combinations(objs.keys(), num_objs))

    # map_counter = 0
    # while map_counter < num_maps:
        # dictionary of {"0_0": lat/lon tuple}
        # where "0_0" represents map name (object and robot counters)
        # idx_to_robot_pt = {}
        # idx_to_obj_pt = {}
        # idx_to_obj2_pt = {}
        # idx_to_obj3_pt = {}

    for combo in obj_combos:
        map_counter = 0
        obj_coords = {}
        for obj in combo:
            obj_coords[obj] = {}

        while map_counter < num_maps:
            output_dir = city_name
            fig, ax = plt.subplots()
            im = plt.imread("map_images/" + city_name + "_diam_75m.png")
            pixel_width = im.shape[0]
            pixel_height = im.shape[1]
            implot = plt.imshow(im)
            print("map counter: ", map_counter)
            ### Removing robot from june pilot maps
            # robot_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
            # robot_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
            # robot_pt = Point(robot_rand_lon, robot_rand_lat)
            # obj_pt = geometry.Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            # if robot_pt.within(poly):
            for obj in combo:
                output_dir += "_" + obj
                obj_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
                obj_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
                obj_pt = Point(obj_rand_lon, obj_rand_lat)

                if obj_pt.within(poly):
                    # lat/lon to pixel conversion
                    obj_lon = ((obj_pt.x - west) / width) * pixel_width
                    obj_lat = (abs(obj_pt.y - north) / height) * pixel_height
                    ax.scatter([obj_lon], [obj_lat], s=0)
                    obj_image = OffsetImage(plt.imread(objs[obj][0]), zoom=objs[obj][1])
                    ab = AnnotationBbox(obj_image, (obj_lon, obj_lat), frameon=False)
                    ax.add_artist(ab)

                    # TODO: is this supposed to be obj_pt.x , obj_pt.y
                    obj_coords[obj][map_counter] = (obj_pt.y, obj_pt.x)

            if not os.path.exists("pilot_maps/" + output_dir):
                os.makedirs("pilot_maps/" + output_dir)

            plt.savefig("pilot_maps/" + output_dir + "/" + output_dir + "_" + str(map_counter) + ".png", dpi=600, bbox_inches="tight")
            plt.clf()
            map_counter += 1
        # save object coordinate dictionaries for this object combination
        for obj_name in obj_coords:
            with open("pilot_maps/" + output_dir + "/" + obj_name + "_coords.json", "w") as fout:
                json.dump(obj_coords[obj_name], fout)

            # with open("pilot_maps/houston_rcar_gcar/gcar_coords.json", "w") as fout:
            #     json.dump(obj2_pt_dict, fout)

        idx_to_cell_json = json.dumps(idx_to_cell, default=jdefault)
        name_to_idx_json = json.dumps(name_to_idx, default=jdefault)

        idx_to_cell_file = open("pilot_maps/" + output_dir + "/idx_to_cell.json", "w")
        idx_to_cell_file.write(idx_to_cell_json)
        idx_to_cell_file.close()

        name_to_idx_file = open("pilot_maps/" + output_dir + "/name_to_idx.json", "w")
        name_to_idx_file.write(str(name_to_idx_json))
        name_to_idx_file.close()

            # obj_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
            # obj_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
            # obj_pt = Point(obj_rand_lon, obj_rand_lat)
            #
            # obj2_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
            # obj2_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
            # obj2_pt = Point(obj2_rand_lon, obj2_rand_lat)
            #
            # obj3_rand_lon = random.uniform(west + 0.0001, east - 0.0001)
            # obj3_rand_lat = random.uniform(north - 0.0001, south + 0.0001)
            # obj3_pt = Point(obj3_rand_lon, obj3_rand_lat)
            # if obj_pt.within(poly):
            #     fig, ax = plt.subplots()
            #     im = plt.imread("map_images/" + CITY_NAME + "_diam_75m.png")
            #     pixel_width = im.shape[0]
            #     pixel_height = im.shape[1]
            #     implot = plt.imshow(im)
            #
            #     # lat/lon to pixel conversion
            #     obj_lon = ((obj_pt.x - west) / width) * pixel_width
            #     obj_lat = (abs(obj_pt.y - north) / height) * pixel_height
            #     ax.scatter([obj_lon], [obj_lat], s=0)
            #     ab = AnnotationBbox(red_honda, (obj_lon, obj_lat), frameon=False)
            #     ax.add_artist(ab)
            #     # obj_plt = plt.scatter([obj_lon], [obj_lat], s=100, c='g')
            #
            #     obj2_lon = ((obj2_pt.x - west) / width) * pixel_width
            #     obj2_lat = (abs(obj2_pt.y - north) / height) * pixel_height
            #     ax.scatter([obj2_lon], [obj2_lat], s=0)
            #     ab2 = AnnotationBbox(green_toyota, (obj2_lon, obj2_lat), frameon=False)
            #     ax.add_artist(ab2)
                # obj2_plt = plt.scatter([obj2_lon], [obj2_lat], s=100, c='b', marker='*')

                # obj3_lon = ((obj3_pt.x - west) / width) * pixel_size
                # obj3_lat = (abs(obj3_pt.y - north) / height) * pixel_size
                # obj3_plt = plt.scatter([obj3_lon], [obj3_lat], s=100, c='y', marker='d')

                # robot_lon = ((robot_pt.x - west) / width) * pixel_size
                # robot_lat = (abs(robot_pt.y - north) / height) * pixel_size
                # robot_plt = plt.scatter([robot_lon], [robot_lat], s=100, c='r', marker='^')
                # plt.savefig("pilot_maps/houston_rcar_gcar/houston_rcar_gcar_" + str(robot_counter) + "_" + str(obj_counter) + ".png", dpi=600, bbox_inches="tight")
                # plt.savefig("/home/matthew/spatial_lang/spatial-lang/houston_3obj/houston_3obj_" + str(robot_counter) + "_" + str(obj_counter) + ".png", dpi=600)
                # obj_plt.remove()
                # obj2_plt.remove()
                # obj3_plt.remove()
                # robot_plt.remove()
                # obj_counter += 1
                # idx_to_robot_pt[str(robot_counter) + "_" + str(obj_counter)] = (robot_pt.y, robot_pt.x)
                # idx_to_obj_pt[str(robot_counter) + "_" + str(obj_counter)] = (obj_pt.y, obj_pt.x)
                # idx_to_obj2_pt[str(robot_counter) + "_" + str(obj_counter)] = (obj2_pt.y, obj2_pt.x)
                # idx_to_obj3_pt[str(robot_counter) + "_" + str(obj_counter)] = (obj3_pt.y, obj3_pt.x)
    # return idx_to_robot_pt, idx_to_obj_pt, idx_to_obj2_pt, idx_to_obj3_pt
    # return obj_coords

if __name__ == "__main__":
    bbox, idx_to_cell, name_to_idx = scrape_osm(CITY_NAME)
    generate_map_images(bbox, 3, 2, CITY_NAME, idx_to_cell, name_to_idx)
    # robot_pt_dict, obj_pt_dict, obj2_pt_dict, obj3_pt_dict = random_points_plot(bbox, 10, 1)

    # with open("/home/matthew/spatial_lang/spatial-lang/nyc_3obj/robot_coords_nyc_3obj.json", "w") as fout:
    #     json.dump(robot_pt_dict, fout)

    # with open("/pilot_maps/nyc_rcar_gcar/obj_coords_2obj.json", "w") as fout:
    #     json.dump(obj_pt_dict, fout)
    #
    # with open("/home/matthew/spatial_lang/spatial-lang/nyc_3obj/obj2_coords_nyc_3obj.json", "w") as fout:
    #     json.dump(obj2_pt_dict, fout)

    # with open("/home/matthew/spatial_lang/spatial-lang/nyc_3obj/obj3_coords_nyc_3obj.json", "w") as fout:
    #     json.dump(obj3_pt_dict, fout)

    # with open("pilot_maps/houston_rcar_gcar/rcar_coords.json", "w") as fout:
    #     json.dump(obj_pt_dict, fout)
    #
    # with open("pilot_maps/houston_rcar_gcar/gcar_coords.json", "w") as fout:
    #     json.dump(obj2_pt_dict, fout)
    #
    # idx_to_cell_json = json.dumps(idx_to_cell, default=jdefault)
    # name_to_idx_json = json.dumps(name_to_idx, default=jdefault)
    #
    # idx_to_cell_file = open("pilot_maps/houston_rcar_gcar/idx_to_cell.json", "w")
    # idx_to_cell_file.write(idx_to_cell_json)
    # idx_to_cell_file.close()
    #
    # name_to_idx_file = open("pilot_maps/houston_rcar_gcar/name_to_idx.json", "w")
    # name_to_idx_file.write(str(name_to_idx_json))
    # name_to_idx_file.close()


    ### TODO ###
    # utf-8 problems
    # not handling the possibility of multiple places with same name
    # visualization?
