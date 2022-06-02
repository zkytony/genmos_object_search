#######################################################
#           Lat/Long Support Functions                #
#######################################################
import math
import requests
import geolocation
import yaml
import json
from shapely.geometry import Polygon, LineString

def get_lat(p):
    assert(isinstance(p, tuple)), ("%s datatype not supported as lat/long datatype" % type(p))
    return p[0]

def get_lon(p):
    assert(isinstance(p, tuple)), ("%s datatype not supported as lat/long datatype" % type(p))
    return p[1]

# Calculate great-circle distance between two points
# Credit: https://www.movable-type.co.uk/scripts/latlong.html
def haversine_distance(p1, p2):
    r = 6371e3
    print("p1", p1)
    print("p2", p2)
    del_lat = abs(math.radians(get_lat(p2) - get_lat(p1)))
    del_lon = abs(math.radians(get_lon(p2) - get_lon(p1)))

    a = math.sin(del_lat / 2) * math.sin(del_lat / 2) + math.cos(get_lat(p1)) * math.cos(get_lat(p2)) * math.sin(del_lon / 2) * math.sin(del_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = r * c

    return d

# Returns SW, NE lat/long points x meters from p
# Credit: https://gis.stackexchange.com/questions/15545/calculating-coordinates-of-square-x-miles-from-center-point
def gdiag(r, p):
    NS = r / 69
    EW = NS / math.cos(get_lat(p))

    d1 = (get_lat(p) - NS, get_lon(p) + EW)
    d2 = (get_lat(p) + NS, get_lon(p) - EW)

    return d1, d2
#    return (round(d1[0], 6), round(d1[1], 6)), ((round(d2[0], 6)), round(d2[1], 6))

def gdiag_inkm(r, p):
    NS = r / 111.2
    EW = abs(math.cos(p[0] * (math.pi / 180)))

    d1 = (get_lat(p) - NS, get_lon(p) + EW)
    d2 = (get_lat(p) + NS, get_lon(p) - EW)
    return d1, d2

# Scrape box of landmarks around p
def create_map_from_osm(p, radius):
    overpass_url = "http://overpass-api.de/api/interpreter"

    # left_bound, right_bound = gdiag(meters_to_miles(radius), p)
    nw, right_bound, left_bound, se = bbox(p, radius)
    print("left bound:", left_bound)
    print("right bound:", right_bound)
    # Visualization query
    overpass_query = """
    [out:json];
    (node["name"]( {}, {}, {}, {});
    way({}, {}, {}, {});
    );
    (._;>;);
    out body;
    """.format(left_bound[0], left_bound[1], right_bound[0], right_bound[1], left_bound[0], left_bound[1], right_bound[0], right_bound[1])


    # Overpass query gets all named nodes and named ways that aren't too large
    # [!] tags exclude large ways
    '''
    overpass_query = """
    [out:json];
    (node["name"]( {}, {}, {}, {});
    way["name"!="Brown University"]["name"][!"place"][!"highway"][!"railway"][!"waterway"][!"boundary"]({}, {}, {}, {});
    );
    (._;>;);
    out body;
    """.format(left_bound[0], left_bound[1], right_bound[0], right_bound[1], left_bound[0], left_bound[1], right_bound[0], right_bound[1])
    '''
    # print("QUERY:", overpass_query)

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data

def bbox(p, radius):
    loc = geolocation.GeoLocation.from_degrees(p[0], p[1])
    ne, nw, sw, se = loc.bounding_locations(radius / float(1000.0))
    return nw, ne, sw, se

def bbox_cell(p, diam):
    """
    p is a tuple (lat, lon) that represents the northwest corner of a cell
    diam is the diameter in meters from the northwest corner to the southeast corner
    return the northwest, northeast, southwest, and southeast corners of the cell
        as tuples of (lat, lon)
    """
    loc = geolocation.GeoLocation.from_degrees(p[0], p[1])
    ne, nw, sw, se = loc.bounding_locations(diam / float(1000.0))
    # p is northwest corner
    return p, (p[0], ne[1]), (se[0], p[1]), se

def meters_to_miles(mt):
    return mt / 1609.344

def calc_axis(polygon):
    """
    Taken from https://stackoverflow.com/questions/13536209/python-efficient-way-to-measure-region-properties-using-shapely/52173616
    """
    line_dict = {}
    mbr_points = list(zip(*polygon.minimum_rotated_rectangle.exterior.coords.xy))
    # calculate the length of each side of the minimum bounding rectangle
    for i in range(len(mbr_points) - 1):
        line_dict[LineString((mbr_points[i], mbr_points[i+1])).length] = LineString((mbr_points[i], mbr_points[i+1]))

    # get major/minor axis measurements
    major_axis = list(line_dict[max(line_dict)].coords)
    minor_axis = list(line_dict[min(line_dict)].coords)
    # 
    # print("_______")
    # print("major is ", major_axis)
    # print("minor is ", minor_axis)
    # print("area is: ", polygon.area)
    # print("length is: ", polygon.length)

    return major_axis, minor_axis
