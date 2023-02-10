# OSM scripts

Code used to create OSM maps. This code is for reference only. Please directly use the `SL_OSM_Maps` dataset.

## How to create test maps for a certain city:
Run multi_obj_vis.py. On lines 67-101 you can choose a city to create a map for. Or, make a new one by providing a new lat/lon center point. By default, the script will produce three object maps, but you can make it two objects by commenting out anything that has to do with the third object. On line 270, you can control how many maps you want by modifying the robot_num and obj_num parameters. This will also produce files that state the idx_to_cell (mapping of index numbers to GridCells in the map), name_to_idx (mapping of landmark names to the cell indices that they occupy), and object coordinates (the lat/lon of the target objects for each map).

If you want to change the object combinations, you'll have to manually change the images used in the AnnotationBbox calls and the name of the output files (like "city_bike_rcar.json").

(I agree that this isn't the cleanest way to do things -- eventually I should rework the code so you can pass in the city center, number of objects, and type of objects as arguments.)

## Scripts
geolocation.py: Helper script for dealing with some lat/lon functions.

grid.py: Scrape OSM data for a map and create the idx_to_cell and name_to_idx files for that map. Does not create visualizations (no pngs).

grid_visualization.py: Create one object maps for a city.

latlon_to_cell_idx.py: The target object coordinates are stored in lat/lon format, but maybe you want to know the cell index they belong to. Using idx_to_cell and the object coord file, produce a mapping of map index to the cell index of the specified target object in that map.

map_to_string.py: Using the idx_to_cell file and the object coords for a map, translate a map into a POMDP world string.

multi_obj_vis.py: Create multi object maps for a city.

parse_osm_mapping.py: Scrape data from OSM and return named ways, building ways, and named nodes.

utils.py: Some helper functions for lat/lon stuff.
