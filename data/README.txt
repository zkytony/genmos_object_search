data

SL_OSM_Dataset/ is the original dataset collected for the SLOOP paper.

robot_tests/ contains maps built when testing out sloop_ros on real robots.
             Note that the map organization is DIFFERENT FROM SL_OSM_Dataset.
             Also, it ONLY STORES LANDMARK information, not the grid map itself.
             So, because one grid map is (supposed to be) fixed for one point cloud,
             you should name each point cloud map differently.
