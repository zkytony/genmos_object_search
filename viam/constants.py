SEARCH_SPACE_RESOLUTION_3D = 0.1

DETECTION2D_CONFIDENCE_THRES = 0.5

#################################################
# The following configurations are specific to
# the UR5 robot arm at Viam as of 12/12/2022 15:44
#################################################
ARM = "arm"
# The joint positions of UR5's home configuration
UR5_HOME_CONFIG = [0.0015838055590579134, -0.001110828108109139,
                   0.0020936803516948656, 0.0017032098299361144,
                   -0.0022029633784150697, 0.0020046605213987424]
COLOR_CAM = "gripper-main:color-cam"
DEPTH_CAM = "gripper-main:depth-corrected"
COMB_CAM = "gripper-main:comb-cam"
DETECTOR = "find_objects"
