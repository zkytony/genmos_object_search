import rospy
from sloop_ros.msg import GridMap2d, GridMapLoc
from sloop_object_search.oopomdp.models.grid_map import GridMap

def grid_map_to_ros_msg(grid_map, stamp=None):
    """
    Given grid_map (GridMap), return a GridMap2d message
    """
    if stamp is None:
        stamp = rospy.Time.now()
    grid_map_msg = GridMap2d()
    grid_map_msg.stamp = stamp
    grid_map_msg.width = grid_map.width
    grid_map_msg.length = grid_map.length
    grid_map_msg.name = grid_map.name
    grid_map_msg.metric_gx_min = grid_map.ranges_in_metric[0][0]
    grid_map_msg.metric_gx_max = grid_map.ranges_in_metric[0][1]
    grid_map_msg.metric_gy_min = grid_map.ranges_in_metric[1][0]
    grid_map_msg.metric_gy_max = grid_map.ranges_in_metric[1][1]
    grid_map_msg.grid_size = grid_map.grid_size
    # We just keep obstacles and free spaces
    locations = []
    for pos in grid_map.obstacles:
        locmsg = GridMapLoc()
        locmsg.loc.x = pos[0]
        locmsg.loc.y = pos[1]
        locmsg.type = "obstacle"
        locmsg.label = grid_map.labels.get(pos, "")
        locations.append(locmsg)
    for pos in grid_map.free_locations:
        locmsg = GridMapLoc()
        locmsg.loc.x = pos[0]
        locmsg.loc.y = pos[1]
        locmsg.type = "free"
        locmsg.label = grid_map.labels.get(pos, "")
        locations.append(locmsg)
    grid_map_msg.locations = locations
    return grid_map_msg


def ros_msg_to_grid_map(grid_map_msg):
    obstacles = set((locmsg.loc.x, locmsg.loc.y) for locmsg in grid_map_msg.locations
                    if locmsg.type == "obstacle")
    free_locations = set((locmsg.loc.x, locmsg.loc.y) for locmsg in grid_map_msg.locations
                         if locmsg.type == "free")
    labels = {(locmsg.loc.x, locmsg.loc.y): locmsg.label
              for locmsg in grid_map_msg.locations}
    ranges_in_metric = ((grid_map_msg.metric_gx_min, grid_map_msg.metric_gx_max),
                        (grid_map_msg.metric_gy_min, grid_map_msg.metric_gy_max))
    return GridMap(grid_map_msg.width,
                   grid_map_msg.length,
                   obstacles,
                   free_locations=free_locations,
                   name=grid_map_msg.name,
                   grid_size=grid_map_msg.grid_size,
                   ranges_in_metric=ranges_in_metric,
                   labels=labels)
