# Visualization of a MOS instance using pygame
#
# Note to run this file, you need to run the following
# in the parent directory of multi_object_search:
#
#   python -m multi_object_search.env.visual
#

# To import cv2 properly, need to remove and append back ros's python packages
import sys
ROS_PKG_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
_has_ros = ROS_PKG_PATH in sys.path
if _has_ros:
    sys.path.remove(ROS_PKG_PATH)  # in order to import cv2 under python3
import cv2
if _has_ros:
    sys.path.append(ROS_PKG_PATH)  # append back in order to import rospy

import pygame
import math
import numpy as np
import random

from pomdp_py import util
from .env import *
from ..domain.observation import *
from ..domain.action import *
from ..domain.state import *
from ..example_worlds import *

#### Visualization through pygame ####
class MosViz:

    def __init__(self, env,
                 res=30, fps=30, controllable=False,
                 bg_path=None, forefs=[]):
        """
        Args:
            env (MosEnvironment): The MOS Environment
            res (int): The resolution, i.e. the amount of pixels
                side of grid cell. This means the size of the
                displayed window has size dimensions of size
                (env.width*res, env.height*res)
            fps (int): Frames-per-second refresh rate of pygame.
            bg_path (str): Path to an image to be used as background
                (e.g. an image of a map). The image will be rescaled
                to fit the width and length of the search space scaled
                by the resolution.
            forefs (list): List of (landmark, keyword, foref) tuples,
                where foref is a tuple (x, y, th) that represents the frame
                of reference for the spatial keyword given landmark. Supply
                to visualize forefs.
        """
        self._env = env
        self._res = res
        self._last_observation = {}  # map from robot id to MosOOObservation
        self._last_viz_observation = {}  # map from robot id to MosOOObservation
        self._last_action = {}  # map from robot id to Action
        self._last_belief = {}  # map from robot id to OOBelief
        self._forefs = forefs

        self._controllable = controllable
        self._running = False
        self._fps = fps
        self._playtime = 0.0

        # Generate some colors, one per target object
        colors = {
            12: (50, 168, 82),  # Toyota (green)
            23: (247, 15, 150),  # Bike (pink)
            34: (191, 17, 52)  # Honda (red)
        }
        self._target_colors = colors
        self._img = self._make_gridworld_image(res, bg_path)
        self._img_history = []


    def _make_gridworld_image(self, r, bg_path=None):
        return MosViz.make_gridworld_image(self._env.width,
                                           self._env.length,
                                           r, bg_path=bg_path,
                                           state=self._env.state,
                                           target_colors=self._target_colors)

    @classmethod
    def highlight_gridcells(cls, img, coords, res,
                            color=(255, 107, 107), alpha=None):
        img_paint = img
        if alpha is not None:
            img_paint = img.copy()
        radius = int(round(res / 2))
        for x,y in coords:
            cv2.circle(img_paint, (y*res + radius,
                                   x*res + radius), radius, color, thickness=-1)
        if alpha is not None:
            img = cv2.addWeighted(img_paint, alpha, img, 1 - alpha, 0)
        return img

    @classmethod
    def make_gridworld_image(cls, width, length, res,
                             state=None, bg_path=None, target_colors={}):
        """Returns an opencv image of the gridworld. If `state` is given,
        then the grid cells corresponding to the robot, obstacles, and
        target objects will be marked. If `bg_path` is provided, then
        a background image will be displayed."""
        # convenience
        w,l,r = width,length,res
        # Preparing 2d array
        arr2d = np.full((width, length), 0)  # free grids
        if state is not None:
            for objid in state.object_states:
                pose = state.object_states[objid]["pose"]
                if state.object_states[objid].objclass == "robot":
                    arr2d[pose[0], pose[1]] = 0  # free grid
                elif state.object_states[objid].objclass == "obstacle":
                    arr2d[pose[0], pose[1]] = 1  # obstacle
                elif state.object_states[objid].objclass == "target":
                    arr2d[pose[0], pose[1]] = objid  # target

        ## Preparing a background image
        if bg_path is not None:
            img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            img = cv2.flip(img, 1)  # flip horizontally
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w*r, l*r))
        else:
            # Creating image
            img = np.full((w*r,l*r,3), 255, dtype=np.int32)

        ## draw the gridworld
        for x in range(w):
            for y in range(l):
                if arr2d[x,y] == 0:    # free
                    if bg_path is None:
                        # If we don't have a background, then plot
                        # a white square indicating free space
                        cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                      (255, 255, 255), -1)
                elif arr2d[x,y] == 1:  # obstacle
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                else:  # target
                    objid = arr2d[x,y]
                    if objid in target_colors:
                        color = target_colors[objid]
                    else:
                        color = (255, 165, 0)
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  color, -1)
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)
        return img

    @property
    def img_width(self):
        return self._img.shape[0]

    @property
    def img_height(self):
        return self._img.shape[1]

    @property
    def gridworld_img(self):
        return self._img

    def update_img(self, img):
        self._img = img

    @property
    def resolution(self):
        return self._res

    @property
    def last_observation(self):
        return self._last_observation

    def record_game_state(self, img):
        self._img_history.append(copy.deepcopy(img))

    @property
    def img_history(self):
        return self._img_history

    def update(self, robot_id, action, observation, viz_observation, belief):
        """
        Update the visualization after there is new real action and observation
        and updated belief.

        Args:
            observation (MosOOObservation): Real observation
            viz_observation (MosOOObservation): An observation used to visualize
                                                the sensing region.
        """
        self._last_action[robot_id] = action
        self._last_observation[robot_id] = observation
        self._last_viz_observation[robot_id] = viz_observation
        self._last_belief[robot_id] = belief

    @staticmethod
    def draw_robot(img, x, y, th, size, color=(255,12,12)):
        radius = int(round(size / 2))
        cv2.circle(img, (y+radius, x+radius), radius, color, thickness=2)

        endpoint = (y+radius + int(round(radius*math.sin(th))),
                    x+radius + int(round(radius*math.cos(th))))
        cv2.line(img, (y+radius,x+radius), endpoint, color, 2)

    @staticmethod
    def draw_observation(img, z, rx, ry, rth, r, size, color=(12,12,255)):
        assert type(z) == MosOOObservation, "%s != MosOOObservation" % (str(type(z)))
        radius = int(round(r / 2))
        for objid in z.objposes:
            if z.for_obj(objid).pose != ObjectObservation.NULL:
                lx, ly = z.for_obj(objid).pose
                cv2.circle(img, (ly*r+radius,
                                 lx*r+radius), size, color, thickness=-1)

    @staticmethod
    def draw_belief(img, belief, r, size, target_colors):
        """belief (OOBelief)"""
        radius = int(round(r / 2))

        circle_drawn = {}  # map from pose to number of times drawn

        for objid in belief.object_beliefs:
            if isinstance(belief.object_belief(objid).random(), RobotState):
                continue
            hist = belief.object_belief(objid).get_histogram()
            if objid in target_colors:
                color = target_colors[objid]
            else:
                color = (255, 165, 0)

            last_val = -1
            count = 0
            for state in reversed(sorted(hist, key=hist.get)):
                if state.objclass == 'target':
                    if last_val != -1:
                        color = util.lighter(color, 1-hist[state]/last_val)
                    if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.999:
                        tx, ty = state['pose']
                        if (tx,ty) not in circle_drawn:
                            circle_drawn[(tx,ty)] = 0
                        circle_drawn[(tx,ty)] += 1

                        cv2.circle(img, (ty*r+radius,
                                         tx*r+radius), size//circle_drawn[(tx,ty)], color, thickness=-1)
                        last_val = hist[state]

                        count +=1
                        if last_val <= 0:
                            break

    # PyGame interface functions
    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def on_event(self, event):
        # TODO: Keyboard control multiple robots
        robot_id = list(self._env.robot_ids)[0]  # Just pick the first one.

        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            u = None  # control signal according to motion model
            action = None  # control input by user

            # odometry model
            if event.key == pygame.K_LEFT:
                action = MoveLeft
            elif event.key == pygame.K_RIGHT:
                action = MoveRight
            elif event.key == pygame.K_UP:
                action = MoveForward
            elif event.key == pygame.K_DOWN:
                action = MoveBackward
            # euclidean axis model
            elif event.key == pygame.K_a:
                action = MoveWest
            elif event.key == pygame.K_d:
                action = MoveEast
            elif event.key == pygame.K_s:
                action = MoveSouth
            elif event.key == pygame.K_w:
                action = MoveNorth
            elif event.key == pygame.K_SPACE:
                action = Look
            elif event.key == pygame.K_RETURN:
                action = Find

            if action is None:
                return

            if self._controllable:
                if isinstance(action, MotionAction):
                    reward = self._env.state_transition(action, execute=True, robot_id=robot_id)
                    z = None
                elif isinstance(action, LookAction) or isinstance(action, FindAction):
                    robot_pose = self._env.state.pose(robot_id)
                    z = self._env.sensors[robot_id].observe(robot_pose,
                                                            self._env.state)
                    self._last_observation[robot_id] = z
                    self._last_viz_observation[robot_id] = z
                    reward = self._env.state_transition(action, execute=True, robot_id=robot_id)
                print("robot state: %s" % str(self._env.state.object_states[robot_id]))
                print("     action: %s" % str(action.name))
                print("     observation: %s" % str(z))
                print("     reward: %s" % str(reward))
                print("------------")
            return action

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0

    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        img = self.render_env(self._display_surf)
        robot_id = list(self._env.robot_ids)[0]  # Just pick the first one.
        rx, ry, rth = self._env.state.pose(robot_id)
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        last_action = self._last_action.get(robot_id, None)
        last_action_str = "no_action" if last_action is None else str(last_action)
        pygame.display.set_caption("%s | Robot%d(%.2f,%.2f,%.2f) | %s | %s" %
                                   (last_action_str, robot_id, rx, ry, rth*180/math.pi,
                                    str(self._env.state.object_states[robot_id]["objects_found"]),
                                    fps_text))
        pygame.display.flip()
        return img

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def render_env(self, display_surf):
        # draw robot, a circle and a vector
        img = np.copy(self._img)
        for i, robot_id in enumerate(self._env.robot_ids):
            rx, ry, rth = self._env.state.pose(robot_id)
            r = self._res  # Not radius!
            last_observation = self._last_observation.get(robot_id, None)
            last_viz_observation = self._last_viz_observation.get(robot_id, None)
            last_belief = self._last_belief.get(robot_id, None)
            if last_belief is not None:
                MosViz.draw_belief(img, last_belief, r, r//3, self._target_colors)
            if last_viz_observation is not None:
                MosViz.draw_observation(img, last_viz_observation,
                                        rx, ry, rth, r, r//4, color=(200, 200, 12))
            if last_observation is not None:
                MosViz.draw_observation(img, last_observation,
                                        rx, ry, rth, r, r//8, color=(20, 20, 180))
            if len(self._forefs) > 0:
                for landmark_, spatial_keyword, foref in self._forefs:
                    img = self.render_foref(img, foref)
            MosViz.draw_robot(img, rx*r, ry*r, rth, r, color=(12, 12, 255*(0.8*(i+1))))
        pygame.surfarray.blit_array(display_surf, img)
        return img

    def render_foref(self, img, foref):
        """foref: (x, y, theta)"""
        x, y, th = foref

        r = self._res
        radius = int(round(r / 4))

        cv2.circle(img, (int(round(y*r))+radius,
                         int(round(x*r))+radius), radius, (255, 145, 145), thickness=-1)
        endpoint = (int(round(y*r)) + int(round(r*math.sin(th))*6),
                    int(round(x*r)) + int(round(r*math.cos(th))*6))
        cv2.arrowedLine(img, (int(round(y)*r)+radius, int(round(x)*r)+radius),
                      endpoint, (255, 145, 145), 10)
        return img


    def update_gridworld_image(self, img):
        if img.shape == self._img.shape:
            self._img = img
        else:
            raise ValueError("Cannot update gridworld image due to dimension mismatch")


def unittest():
    # If you don't want occlusion, use this:
    laserstr = make_laser_sensor(90, (1, 8), 0.5, False)
    # If you want occlusion, use this
    # (the difference is mainly in angle_increment; this
    #  is due to the discretization - discretization may
    #  cause "strange" behavior when checking occlusion
    #  but the model is actually doing the right thing.)
    laserstr_occ = make_laser_sensor(360, (1, 8), 0.5, True)
    # Proximity sensor
    proxstr = make_proximity_sensor(1.5, False)
    proxstr_occ = make_proximity_sensor(1.5, True)

    worldmap, robot = world1
    worldstr = equip_sensors(worldmap, {robot: laserstr})
    env = interpret(worldstr)
    viz = MosViz(env, controllable=True)
    viz.on_execute()

if __name__ == '__main__':
    unittest()
