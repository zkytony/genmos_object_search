from ..parsing.graph.ego import *
import math

nyc_specs = {
    "butterfield_market": {
        "coords": [8, 0],
        "divisions": 8,
        "color": 'red',
        "target": False
    },
    "Lexington_Ave": {
        "coords": [7, 9],
        "divisions": 8,
        "color": 'red',
        "target": False
    },
    "East_77th_Street": {
        "coords": [17, 16],
        "divisions": 8,
        "color": 'red',
        "target": False
    },
    "Duane_Reade": {
        "coords": [6, 17],
        "divisions": 8,
        "color": 'red',
        "target": False
    },
    "Starbucks": {
        "coords": [7, 14],
        "divisions": 8,
        "color": 'red',
        "target": False
    },
    "Second_Chance": {
        "coords": [12, 7],
        "divisions": 8,
        "color": 'red',
        "target": False
    },
    "Dunkin_Donuts": {
        "coords": [7, 5],
        "divisions": 8,
        "color": 'red',
        "target": False
    }
}


def nyc_hardcoded(objects):
    """Outputs ego graphs for:
    The car is on Lexington between Starbucks and Duane.

    The landmarks here are Lexington, Starbucks, and Duane.

    objects (set) a set of object ids"""

    lexington = EgoGraph("Lexington_Ave",
                         objects=objects,
                         divisions=4)
    starbucks = EgoGraph("Starbucks",
                         objects=objects,
                         divisions=8)
    duane = EgoGraph("Duane_Reade",
                     objects=objects,
                     divisions=8)
    eg_list = [lexington, starbucks, duane]
    # Now create groundings, in the order of eg_list.
    lexington_grounding = {
        -1: tuple(nyc_specs[lexington.name]["coords"]),  # center coordinates
        0: 1, # for viewnum 0
        1: 3, # for viewnum 1
        2: 1, # for viewnum 2
        3: 3, # for viewnum 3
        "phase_shift": 30*math.pi / 180.0
    }
    lexington.node_at(1).set_lh(list(objects)[0], 100)
    lexington.node_at(3).set_lh(list(objects)[0], 100)

    starbucks_grounding = {
        -1: tuple(nyc_specs[starbucks.name]["coords"]),  # center coordinates
        "phase_shift": 30*math.pi / 180.0
    }
    starbucks.node_at(1).set_lh(list(objects)[0], 200)
    starbucks.node_at(2).set_lh(list(objects)[0], 200)
    starbucks.node_at(3).set_lh(list(objects)[0], 200)

    duane_grounding = {
        -1: tuple(nyc_specs[duane.name]["coords"]),  # center coordinates
        "phase_shift": 30*math.pi / 180.0
    }
    duane.node_at(5).set_lh(list(objects)[0], 200)
    duane.node_at(6).set_lh(list(objects)[0], 200)
    duane.node_at(7).set_lh(list(objects)[0], 200)

    grounding_specs = [lexington_grounding,
                       starbucks_grounding,
                       duane_grounding]
    return eg_list, grounding_specs
