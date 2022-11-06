import math
import yaml
import numpy as np
import open3d as o3d
from dataclasses import dataclass

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions

from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionServiceClient, VisModelConfig, VisModelType
from viam.services.motion import MotionServiceClient

from viam.proto.common import ResourceName, PoseInFrame

import sloop_object_search.utils.math as math_utils



########### Robot-Specific viam functions ###########
async def connect_viamlab_ur5():
    creds = Credentials(
        type='robot-location-secret',
        payload='gm1rjqe84nt8p64ln6r1jyf5hc3tdnc2jywojoykvk56d0qa')
    opts = RobotClient.Options(
        refresh_interval=0,
        dial_options=DialOptions(credentials=creds)
    )
    return await RobotClient.at_address('viam-test-bot-main.tcyat99x8y.viam.cloud', opts)

########### Robot-generic viam functions ###########
async def viam_get_ee_pose(viam_robot):
    """return current end-effector pose in world
    frame through Viam.
    Return type: tuple (x,y,z,qx,qy,qz,qw)"""
    #NOTE!!! BELOW DOES NOT GIVE YOU THE TRUE EE
    #ON THE GRIPPER OF THE UR5 ROBOT AT VIAM LAB
    #BUT THE END OF THE ARM WITHOUT GRIPPER. THIS
    #IS BECAUSE THE GRIPPER IS A SEPARATE COMPUTER
    #AND CURRENTLY THERE IS A BUG IN VIAM TO GET
    #THAT FRAME. ALSO, SHOULD USE MOTIONSERVICE
    #INSTEAD OF ARM BUT CANT BECAUSE OF THAT BUG.
    arm = Arm.from_robot(viam_robot, "arm")
    pose_w_ovec = await arm.get_end_position()

    # convert pose orientation to quaternion! [viam's problem]
    ovec = OrientationVector(Vector3(
        pose_w_ovec.o_x, pose_w_ovec.o_y, pose_w_ovec.o_z), math_utils.to_rad(pose_w_ovec.theta))
    quat = Quaternion.from_orientation_vector(ovec)
    pose_w_quat = (pose_w_ovec.x, pose_w_ovec.y, pose_w_ovec.z,
                   quat.i, quat.j, quat.k, quat.real)
    return pose_w_quat

async def viam_get_point_cloud_array(viam_robot, target_frame="camera"):
    """return current point cloud from camera through Viam.
    If 'target_frame' == 'camera', then the returned points will be in camera frame. If
    'target_frame' == 'world', then the returned points will be in the world frame.

    Return type: numpy array of [x,y,z]
    """
    camera = Camera.from_robot(viam_robot, "gripper:depth-cam")
    data, mimetype = await camera.get_point_cloud()
    # TODO: a better way?
    with open("/tmp/pointcloud_data.pcd", "wb") as f:
        f.write(data)
    pcd = o3d.io.read_point_cloud("/tmp/pointcloud_data.pcd")
    points_T_camera = np.asarray(pcd.points)
    if target_frame == "camera":
        return points_T_camera
    elif target_frame == "world":
        # TODO: transform points in camera frame to be in world frame,
        # leveraging the camera pose.
        # # Get transform of camera wrt world. This is
        # pose_w_quat = await viam_get_ee_pose(viam_robot)
        # x,y,z,qx,qy,qz,qw = pose_w_quat
        # TODO: Finish this.
        raise NotImplementedError()
    else:
        raise ValueError(f"Unsupported target frame: {target_frame}")

def viam_get_object_detections3d(world_frame):
    """Return type: a list of (label, box3d) tuples.
    A label is a string.
    A box3d is a tuple (center, w, l, h)
    Note that we want 'center' in the world frame. In
    the case of a tabletop robot, it should be the frame
    of its base."""
    # NOTE: SKIPPING THIS BECAUSE WILL NOT USE POINT CLOUD AS
    # THE depth-cam is UNRELIABLE FOR THE UR5 GRIPPER AT VIAM LAB.
    raise NotImplementedError()

async def viam_get_image(viam_robot, camera_name, return_type="PIL"):
    """Returns image from given camera"""
    camera = Camera.from_robot(viam_robot, camera_name)
    # This image should be a PIL image
    image = await camera.get_image()
    if return_type == "PIL":
        return image
    elif return_type == "array":
        imgarr = np.array(image)
        return imgarr
    else:
        raise ValueError(f"Unsupported return type {return_type}")


async def viam_get_object_detections2d(
        viam_robot,
        camera_name="segmenter-cam",
        detector_name="find_objects"):
    """
    Args:
        viam_robot_or_vision_client: either viam_robot connection, or VisionServiceClient
        camera_name (str): name of camera with color image
        detector_name (str): name of RGB object detection
    Returns:
        Return type: a list of (label, box2d) tuples.
        A label is a string.
        box2d is xyxy tuple
    """
    vision_client = VisionServiceClient.from_robot(viam_robot)
    detections = await vision_client.get_detections_from_camera(
        camera_name, detector_name)
    return detections


async def viam_move_ee_to(viam_robot, pos, orien, action_id):
    """
    Moves the end effector to the given goal position and orientation.
    If not possible, [???]

    pos (position): (x,y,z)
    orien (quaternion): (qx, qy, qz, qw)
    """
    pose = await viam_get_ee_pose(viam_robot)
    motion = MotionServiceClient.from_robot(viam_robot)
    # motion.move("arm", )
    pose.x += 100
    motion = MotionServiceClient.from_robot(robot)
    for resname in robot.resource_names:
        if resname.name == "arm":
            print (resname)
            move = await motion.move(component_name=resname, destination = PoseInFrame(reference_frame="arm_origin", pose=pose))
    # arm = Arm.from_robot(viam_robot, "arm")
    # pose = await arm.get_end_position()
    # print(pose)
    # # # prefers to use services methods and not component methods.
    # motion = MotionServiceClient.from_robot(viam_robot)
    # await motion.move(Gripper.get_resource_name("gripper:vg1"),
    #                   PoseInFrame(reference_frame="arm_origin",
    #                               pose=pose))

    for resname in viam_robot.resource_names:
        if resname.name == "gripper:vg1":
            print(resname.name)
            pose = await motion.get_pose(resname, "world")
                                 #                            type="component",
                                 #                            subtype="gripper",
                                 #                            name="gripper:vg1"),
                                 # destination_frame="world")
    print(pose)

    # from viam_utils import OrientationVector, Quaternion, Vector3
    # ovec = OrientationVector(Vector3(pose.o_x, pose.o_y, pose.o_z), math_utils.to_rad(pose.theta))
    # qq = Quaternion.from_orientation_vector(ovec)
    # ovec2 = qq.to_orientation_vector()
    # qq2 = Quaternion.from_orientation_vector(ovec2)

    # print(qq)
    # print(qq2)

    # import pdb; pdb.set_trace()

    # viam represents orientation by ox, oy, oz, theta
    # where (ox, oy, oz) is the axis of rotation, and
    # theta is the degree of rotation. We convert that
    # to quaternion by the definition of quaternion.

    # qx = x * math.sin(math_utils.to_rad(pose.theta) / 2)
    # qy = y * math.sin(math_utils.to_rad(pose.theta) / 2)
    # qx = math.cos(math_utils.to_rad(pose.theta) / 2)
    # pass


def viam_signal_find(action_id):
    """Do something with the robot to signal the find action"""
    raise NotImplementedError

def viam_detections3d_to_proto(robot_id, detections):
    """Parameters:
    detections: a list of (label, box3d) tuples.
    A label is a string.
    A box3d is a tuple (center, w, l, h) -- this is interpreted from Viam's proto def.
    Note that 'center' should already be in world frame.
    """
    detections_pb = []
    for det3d in detections:
        label, box3d = det3d
        center, w, l, h = box3d
        center_pb = proto_utils.posetuple_to_poseproto(center)
        box_pb = common_pb2.Box3D(center=center_pb,
                                  sizes=common_pb2.Vec3(x=w, y=l, z=h))
        # NOTE: setting confidence is not supported right now
        det3d_pb = o_pb2.Detection3D(label=label,
                                     box=box_pb)
        detections_pb.append(det3d_pb)
    # TODO: properly create header
    raise NotImplementedError()
    header = proto_utils.make_header(frame_id=None, stamp=None)
    return o_pb2.ObjectDetectionArray(header=header,
                                      robot_id=robot_id,
                                      detections=detections_pb)


################## Below are code from Gautham for orientation conversion #############
@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def clone(self) -> "Vector3":
        return self.__class__(
            x=self.x,
            y=self.y,
            z=self.z
        )

    @classmethod
    def from_tuple(cls, vec) -> "Vector3":
        return cls(x=vec[0], y=vec[1], z=vec[2])

    @classmethod
    def cross_product(cls, v1, v2) -> "Vector3":
        return cls(
            x=(v1.y * v2.z - v1.z * v2.y),
            y=(v1.z * v2.x - v1.x * v2.z),
            z=(v1.x * v2.y - v1.y * v2.x)
        )

    def cross(self, vec) -> "Vector3":
        return self.cross_product(self, vec)

    @classmethod
    def dot_product(cls, v1, v2) -> float:
        return (
            (v1.x * v2.x) +
            (v1.y * v2.y) +
            (v1.z * v2.z)
        )

    def dot(self, vec) -> float:
        return self.dot_product(self, vec)

    @property
    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    @classmethod
    def cosine_between(cls, v1, v2) -> float:
        return (
            cls.dot_product(v1, v2) / (v1.norm * v2.norm)
        )

    def get_normalized(self) -> "Vector3":
        norm = self.norm
        return self.__class__(
            x=(self.x / norm),
            y=(self.y / norm),
            z=(self.z / norm)
        )

    def __add__(self, other) -> "Vector3":
        return Vector3(
            x=(self.x + other.x),
            y=(self.y + other.y),
            z=(self.x + other.z)
        )

    def __truediv__(self, scalar: float) -> "Vector3":
        return Vector3(
            x=(self.x / scalar),
            y=(self.y / scalar),
            z=(self.x / scalar)
        )

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(
            x=(self.x * scalar),
            y=(self.y * scalar),
            z=(self.x * scalar)
        )

    def __sub__(self, other) -> "Vector3":
        return Vector3(
            x=(self.x - other.x),
            y=(self.y - other.y),
            z=(self.x - other.z)
        )

    def __str__(self) -> str:
        return f"(X: {self.x}, Y: {self.y}, Z: {self.z})"

class AxisAngle:

    def __init__(self, rotation_axis: Vector3, theta: float):
        if math.fabs(theta) > (math.pi * 2):
            raise ValueError(
                "theta for axis angle must be in radians "
                "between -2pi and 2pi"
            )
        self.theta = theta
        self.rotation_axis = rotation_axis.get_normalized()



class OrientationVector:

    def __init__(self, sphere_vec: Vector3, theta: float):
        if math.fabs(theta) > (math.pi * 2):
            raise ValueError("theta must be between -2pi and 2pi")
        self.theta = theta
        self.unit_sphere_vec = sphere_vec.get_normalized()

    def __str__(self) -> str:
        return f"OV: ({self.unit_sphere_vec}, Theta: {self.theta})"



    @classmethod
    def extract_from_pose(cls, pose):
        sphere_vec = Vector3(pose.o_x, pose.o_y, pose.o_z)
        return cls(sphere_vec, pose.theta)



@dataclass
class Quaternion:
    real: float
    i: float
    j: float
    k: float

    ANGLE_ACCEPTANCE = 0.0001

    @classmethod
    def from_tuple(cls, q) -> "Quaternion":
        return cls(
            real=q[0],
            i=q[1],
            j=q[2],
            k=q[3]
        )

    @classmethod
    def from_axis_angle(cls, axis_angle: AxisAngle) -> "Quaternion":
        sinHalfTheta = math.sin(axis_angle.theta / 2)

        real = math.cos(axis_angle.theta / 2)
        i = sinHalfTheta * axis_angle.rotation_axis.x
        j = sinHalfTheta * axis_angle.rotation_axis.y
        k = sinHalfTheta * axis_angle.rotation_axis.z

        return Quaternion(
            real=real,
            i=i,
            j=j,
            k=k
        )

    @classmethod
    def multiply(cls, q0: "Quaternion", q1: "Quaternion") -> "Quaternion":
        real = (q0.real*q1.real) - (q0.i*q1.i) - (q0.j*q1.j) - (q0.k*q1.k)
        i = (q0.real*q1.i) + (q0.i*q1.real) + (q0.j*q1.k) - (q0.k*q1.j)
        j = (q0.real*q1.j) - (q0.i*q1.k) + (q0.j*q1.real) + (q0.k*q1.i)
        k = (q0.real*q1.k) + (q0.i*q1.j) - (q0.j*q1.i) + (q0.k*q1.real)
        return cls(real=real, i=i, j=j, k=k)

    def __mul__(self, q: "Quaternion") -> "Quaternion":
        return self.multiply(self, q)

    @property
    def conjugate(self) -> "Quaternion":
        return self.__class__(
            real=self.real,
            i=-self.i,
            j=-self.j,
            k=-self.k
        )

    @property
    def imaginary_vector(self) -> Vector3:
        return Vector3(
            x=self.i,
            y=self.j,
            z=self.k
        )

    def to_orientation_vector(self) -> OrientationVector:
        x_quat = Quaternion.from_tuple((0, -1, 0, 0))
        z_quat = Quaternion.from_tuple((0, 0, 0, 1))
        conj = self.conjugate
        new_x = (self * x_quat) * conj
        new_z = (self * z_quat) * conj
        oV = Vector3(
            x=new_z.i,
            y=new_z.j,
            z=new_z.k
        )
        theta = self._get_theta_for_orientation_vector(new_x, new_z)
        return OrientationVector(
            sphere_vec=oV,
            theta=theta
        )

    def _get_theta_for_orientation_vector(
        self, new_x: "Quaternion", new_z: "Quaternion"
    ):
        if 1 - math.fabs(new_z.k) > self.ANGLE_ACCEPTANCE:
            new_z_imaginary = new_z.imaginary_vector
            new_x_imaginary = new_x.imaginary_vector
            normal_1 = new_z_imaginary.cross(new_x_imaginary)
            z_axis_imaginary = Vector3.from_tuple((0, 0, 1))
            normal_2 = new_z_imaginary.cross(z_axis_imaginary)
            cos_theta = Vector3.cosine_between(normal_1, normal_2)
            cos_theta = min(cos_theta, 1)
            cos_theta = max(cos_theta, -1)
            theta = math.acos(cos_theta)
            if theta > self.ANGLE_ACCEPTANCE:
                rot_about_x_by_neg_theta = AxisAngle(
                    rotation_axis=new_z_imaginary,
                    theta=-theta
                )
                rot_quat = self.from_axis_angle(rot_about_x_by_neg_theta)
                z_axis = Quaternion.from_tuple((0, 0, 0, 1))
                test_z = (rot_quat * z_axis) * rot_quat.conjugate
                test_z_imaginary = test_z.imaginary_vector
                normal_3 = new_z_imaginary.cross(test_z_imaginary)
                cos_test = normal_1.dot(normal_3) / (
                    normal_1.norm * normal_3.norm)
                if (1 - cos_test) < (
                    self.ANGLE_ACCEPTANCE * self.ANGLE_ACCEPTANCE
                ):
                    theta = -theta
        else:
            denom = new_x.i if new_x.k > 0 else -new_x.i
            theta = -math.atan2(new_x.j, denom)
        return theta

    def __str__(self) -> str:
        return f"(Real: {self.real}, I: {self.i}, J: {self.j}, K: {self.k})"


    @classmethod
    def from_orientation_vector(cls, o_vec: OrientationVector):
        unit_ov = o_vec.unit_sphere_vec
        lat = math.acos(unit_ov.z)
        lon = 0
        theta = o_vec.theta

        if 1 - abs(unit_ov.z) > cls.ANGLE_ACCEPTANCE:
            lon = math.atan2(unit_ov.y, unit_ov.x)

        # convert angles to quat using zyz rotational order
        s_0 = math.sin(lon / 2)
        c_0 = math.cos(lon / 2)

        s_1 = math.sin(lat / 2)
        c_1 = math.cos(lat / 2)

        s_2 = math.sin(theta / 2)
        c_2 = math.cos(theta / 2)

        s = [s_0, s_1, s_2]
        c = [c_0, c_1, c_2]

        real = (c[0]*c[1]*c[2] - s[0]*c[1]*s[2])
        i = (c[0]*s[1]*s[2] - s[0]*s[1]*c[2])
        j = (c[0]*s[1]*c[2] + s[0]*s[1]*s[2])
        k = (s[0]*c[1]*c[2] + c[0]*c[1]*s[2])

        return cls(real=real, i=i, j=j, k=k)

    def apply_rotation(self, v: Vector3) -> Vector3:
        # vec_quat = self.__class__(real=0, i=vec.x, j=vec.y, k=vec.z)
        # conj = self.conjugate
        # rotated = self * vec_quat * conj
        # return Vector3(rotated.i, rotated.j, rotated.k)
        u = self.imaginary_vector
        s = self.real
        return (u * (u.dot(v) * 2)) + (v * ((s * s) - u.dot(u)))\
            + (u.cross(v) * (s * 2))
