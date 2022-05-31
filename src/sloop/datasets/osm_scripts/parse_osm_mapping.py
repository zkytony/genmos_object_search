import json
import yaml

def parse_json(js):
    data = json.dumps(js['elements'])
    y = yaml.load(data)
    return y

nodes = {}
named_nodes = {}
ways = {}
unnamed_buildings = []

def build_ways(y):
    global nodes
    global named_nodes
    global ways
    global unnamed_buildings
    
    nodes = {}
    named_nodes = {}
    ways = {}
    unnamed_buildings = []

    for landmark in y:
        # Creating dict of node id's to lat/long
        if landmark["type"] == "node":
            nodes[landmark["id"]] = (landmark["lat"], landmark["lon"])

        # Processing named landmarks
        if "tags" in landmark and "name" in landmark["tags"]:

            # If landmark is way, use one of its bounding nodes for lat/long
            if landmark["type"] == "way":
                l_name = landmark["tags"]["name"]

                # only create an empty list if landmark not already in dict
                if ways.get(l_name) == None:
                    ways[l_name] = []

                for node_id in landmark["nodes"]:
                    ways[l_name].append((nodes[node_id][0], nodes[node_id][1]))

            if landmark["type"] == "node":
                n_name = landmark["tags"]["name"]
                named_nodes[n_name] = (landmark["lat"], landmark["lon"])

        # Process unnamed building ways:
        if "tags" in landmark:
            if "building" in landmark["tags"] and "nodes" in landmark:
                building = []
                for node_id in landmark["nodes"]:
                    building.append((nodes[node_id][0], nodes[node_id][1]))
                unnamed_buildings.append(building)

def get_ways():
    global ways
    return ways

def get_nodes():
    global named_nodes
    return named_nodes

def get_buildings():
    global unnamed_buildings
    return unnamed_buildings
