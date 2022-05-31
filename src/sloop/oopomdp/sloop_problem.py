# Similar to problem.py, except considering also language input.

import sloop.oopomdp.problem as mos
import sloop.oopomdp.lang2obs as l2o
from sloop.oopomdp.landmark_specs import nyc_hardcoded
from sloop.oopomdp.env.env import *
from sloop.oopomdp.example_worlds import *
import sloop.parsing.graph.util as util
from sloop.datasets.SL_OSM_Dataset.mapinfo.constants import FILEPATHS
import numpy as np
import cv2

def sloop_solve(lang,
                oopomdp,
                planner,
                viz=None,
                grounding_specs=[], # specifies grounding parameters (see
                                    # egos_to_obs) todo: right?
                default_dist=1,
                max_time=120,  # maximum amount of time allowed to solve the problem
                max_steps=500): # maximum number of planning steps the agent can take.
    objects = list(oopomdp.agent.belief.object_beliefs.keys())
    try:
        # language to graph
        sg = l2o.splang_to_spgraph(lang)

        # spatial graph to ego graphs
        eg_list = l2o.spgraph_to_egos(oopomdp.agent.grid_map)
    except NotImplementedError:
        print("splang_to_spgraph and/or spgraph_to_egos is Not Yet Implemented."\
              "Running a Mock Trial for now (i.e. hard coded eg_list")
        print("Hard coding the ego graphs for:")
        print("   The green car is on Lexington between Starbucks and Duane.")
        eg_list, grounding_specs = nyc_hardcoded(objects)

    # ego graphs to beliefs
    oo_spl_obsrv, geg_list = l2o.egos_to_obs(objects,
                                             oopomdp.agent.grid_map,
                                             eg_list, specs=grounding_specs,
                                             default_dist=default_dist)

    # Update agent belief using this observation
    l2o.splang_matrix_belief_update(oo_spl_obsrv, oopomdp.agent)

    # visualize belief as result of this update
    viz.update(oopomdp.agent.robot_id,
               None,
               None,
               None,
               oopomdp.agent.cur_belief)

    # Do everything else the same way
    mos.solve(oopomdp, planner, viz=viz,
              max_time=max_time,
              max_steps=max_steps)
              # step_func=sloop_step_func,
              # step_func_args={"geg_list": geg_list})


def sloop_step_func(oopomdp, real_action,
                    real_observation, reward, viz,
                    geg_list=[]):
    # We just care about visualizing the graph
    if viz is None:
        return

    img = np.copy(viz.gridworld_img)

    # all circles have the same size & color
    r = viz.resolution  # resolution
    radius = int(round(r * 0.5 / 2))
    color = (120, 30, 215)

    for geg in geg_list:
        # draw a circle for the center
        cx, cy = geg.coords_by_viewnum(-1)
        print(" Center of %s at %s" % (geg.name, str((cx,cy))))
        cv2.circle(img,
                   (int(round(cy*r + r//2)),
                    int(round(cx*r + r//2))),
                   radius, color,
                   thickness=-1)  # filled

        grad_colors =\
            util.linear_color_gradient(color,
                                       (255-color[0], 255-color[1], 255-color[2]),
                                       len(geg.nodes))
        for i, nid in enumerate(geg.nodes):
            nx, ny = geg.coords_by_id(nid)
            print(" Node %d of %s at %s" % (nid, geg.name, str((nx,ny))))
            # draw a circle for this node
            cv2.circle(img,
                       (int(round(ny*r + r//2)),
                        int(round(nx*r + r//2))),
                       radius, grad_colors[i],
                       thickness=-1)  # filled

            # draw an edge
            cv2.line(img,
                     (int(round(cy*r + r//2)), int(round(cx*r + r//2))),
                     (int(round(ny*r + r//2)), int(round(nx*r + r//2))),
                     grad_colors[i],
                     thickness=3)
    viz.update_gridworld_image(img)
    viz.on_loop()
    viz.on_render()


def unittest():
    bg_path=FILEPATHS["nyc"]["map_png"]
    grid_map, robot_char = world_becky #random_world(14, 14, 3, 5)
    laserstr = make_laser_sensor(90, (1, 5), 0.5, False)
    proxstr = make_proximity_sensor(3, False)
    oopomdp = mos.MosOOPOMDP(robot_char,  # r is the robot character
                             sigma=0.01,  # observation model parameter
                             epsilon=1.0, # observation model parameter
                             grid_map=grid_map,
                             sensors={robot_char: proxstr},
                             prior="uniform",
                             agent_has_map=True,
                             no_look=False,
                             reward_small=1)
    # THIS EXAMPLE IS HARD CODED.
    lang = "The green car is on Lexington between Starbucks and Duane"
    planner, viz = mos.setup_solve(oopomdp,
                                   max_depth=30,  # planning horizon
                                   discount_factor=0.95,
                                   planning_time=1.0,       # amount of time (s) to plan each step
                                   exploration_const=1000, # exploration constant
                                   visualize=True,
                                   bg_path=bg_path)

    import time; time.sleep(1)
    sloop_solve(lang,
                oopomdp,
                planner,
                viz=viz,
                default_dist=1,
                max_time=120,
                max_steps=500)

if __name__ == "__main__":
    unittest()
