import json
import sys
import math

# read in information about a map
# output in string readable format for pomdp

IDX_TO_CELL = "/home/matthew/spatial_lang/spatial-lang/nyc_3obj/idx_to_cell_nyc_3obj.json"
OBJ_CELLS = "/home/matthew/spatial_lang/spatial-lang/nyc_3obj/obj_cells_nyc_3obj.json"
OBJ2_CELLS = "/home/matthew/spatial_lang/spatial-lang/nyc_3obj/obj2_cells_nyc_3obj.json"
OBJ3_CELLS = "/home/matthew/spatial_lang/spatial-lang/nyc_3obj/obj3_cells_nyc_3obj.json"
ROBOT_CELLS = "/home/matthew/spatial_lang/spatial-lang/nyc_3obj/robot_cells_nyc_3obj.json"


# TODO: Do we need a final newline on the last row?
# TODO: Remember to change things for 3obj v 3obj!
# TODO: Remember that the old data collection had txt, not json

with open(IDX_TO_CELL, 'r') as fin:
    idx_to_cell = json.load(fin)

with open(OBJ_CELLS, 'r') as fin:
    obj_cells = json.load(fin)

with open(OBJ2_CELLS, 'r') as fin:
    obj2_cells = json.load(fin)

with open(OBJ3_CELLS, 'r') as fin:
    obj3_cells = json.load(fin)

with open(ROBOT_CELLS, 'r') as fin:
    robot_cells = json.load(fin)

def longest_row():
    """
    Identify the number of cells in the longest row of the map. 
    A new row starts when the north latitude changes (ratio < 1). 
    """
    row_counter = 0
    row_max = 0
    for i in range(len(idx_to_cell) - 1):
        n = idx_to_cell[str(i)]["nw"][0]
        n1 = idx_to_cell[str(i+1)]["nw"][0]
        # check if same row 
        if (n1 / n) == 1:
            row_counter += 1
        else: # new row
            row_max = max(row_max, row_counter)
            row_counter = 0
    return row_max

def identify_cell(i, obj_cell, obj2_cell, obj3_cell, robot_cell):
    """
    Return the string representation for this cell. 
    """
    if i == obj_cell or i == obj2_cell or i == obj3_cell:
        return "T"
    elif i == robot_cell:
        return "r"
    else:
        return "."

    
def main(map_names):
    """
    Converts map into string readable format for py-pomdp
    
    Arguments:
     - map_names: a list of map indices such as ["0_10", "1_5"]

    Returns:
     - nothing, writes map string out to a txt file
    """
    row_len = longest_row() 

    for map_name in map_names:
        obj_cell = obj_cells[map_name]
        obj2_cell = obj2_cells[map_name]
        # obj2_cell = None
        obj3_cell = obj3_cells[map_name]
        # obj3_cell = None
        robot_cell = robot_cells[map_name]

        pomdp_cell_to_map_cell = {}
        map_string = ""
        row_len_counter = 0
        row_idx_counter = 0

        map_string += identify_cell(0, obj_cell, obj2_cell, obj3_cell, robot_cell)
        pomdp_cell_to_map_cell[str((row_len_counter, row_idx_counter))] = 0
        row_len_counter += 1
        for i in range(1, len(idx_to_cell)):
            n = idx_to_cell[str(i-1)]["nw"][0]
            n1 = idx_to_cell[str(i)]["nw"][0]
            ratio = n1 / n
            if ratio < 1:
                if (row_len_counter - 1) < row_len:
                    map_string += "x" * (row_len - row_len_counter + 1)
                map_string += "\n" + identify_cell(i, obj_cell, obj2_cell, obj3_cell, robot_cell)
                row_idx_counter += 1
                pomdp_cell_to_map_cell[str((0, row_idx_counter))] = i
                row_len_counter = 1
            else:
                map_string += identify_cell(i, obj_cell, obj2_cell, obj3_cell, robot_cell)
                pomdp_cell_to_map_cell[str((row_len_counter, row_idx_counter))] = i
                row_len_counter += 1


        with open("nyc_3obj/" + map_name + "_string.txt", 'w') as fout:
            fout.write(map_string)

        # for key in sorted(pomdp_cell_to_map_cell.keys()):
            # print(key, pomdp_cell_to_map_cell[key])

        with open("nyc_3obj/" + map_name + "_pomdp_cell_to_map_idx.json", 'w') as fout:
            json.dump(pomdp_cell_to_map_cell, fout) 

    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please enter at least one map name, such as 0_1")
    else:
        map_names = sys.argv[1:]
        main(map_names)
