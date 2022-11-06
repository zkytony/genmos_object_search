# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sloop_object_search/grpc/observation.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from sloop_object_search.grpc import common_pb2 as sloop__object__search_dot_grpc_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*sloop_object_search/grpc/observation.proto\x12\x18sloop_object_search.grpc\x1a%sloop_object_search/grpc/common.proto\"\xd3\x01\n\rOccupancyGrid\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12;\n\x05grids\x18\x03 \x03(\x0b\x32,.sloop_object_search.grpc.OccupancyGrid.Grid\x1a\x41\n\x04Grid\x12+\n\x03pos\x18\x01 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec2\x12\x0c\n\x04type\x18\x02 \x01(\t\"\xc2\x01\n\nPointCloud\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12:\n\x06points\x18\x03 \x03(\x0b\x32*.sloop_object_search.grpc.PointCloud.Point\x1a\x34\n\x05Point\x12+\n\x03pos\x18\x01 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec3\"\x9b\x01\n\tDetection\x12\r\n\x05label\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\x31\n\x06\x62ox_3d\x18\x03 \x01(\x0b\x32\x1f.sloop_object_search.grpc.Box3DH\x00\x12\x31\n\x06\x62ox_2d\x18\x04 \x01(\x0b\x32\x1f.sloop_object_search.grpc.Box2DH\x00\x42\x05\n\x03\x62ox\"\x93\x01\n\x14ObjectDetectionArray\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x37\n\ndetections\x18\x03 \x03(\x0b\x32#.sloop_object_search.grpc.Detection\"j\n\x08Language\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0c\n\x04text\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\"\xd5\x01\n\tRobotPose\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x33\n\x07pose_2d\x18\x03 \x01(\x0b\x32 .sloop_object_search.grpc.Pose2DH\x00\x12\x33\n\x07pose_3d\x18\x04 \x01(\x0b\x32 .sloop_object_search.grpc.Pose3DH\x00\x12\x12\n\ncovariance\x18\x05 \x03(\x01\x42\x06\n\x04pose\"f\n\x0cObjectsFound\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x12\n\nobject_ids\x18\x03 \x03(\tb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sloop_object_search.grpc.observation_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OCCUPANCYGRID._serialized_start=112
  _OCCUPANCYGRID._serialized_end=323
  _OCCUPANCYGRID_GRID._serialized_start=258
  _OCCUPANCYGRID_GRID._serialized_end=323
  _POINTCLOUD._serialized_start=326
  _POINTCLOUD._serialized_end=520
  _POINTCLOUD_POINT._serialized_start=468
  _POINTCLOUD_POINT._serialized_end=520
  _DETECTION._serialized_start=523
  _DETECTION._serialized_end=678
  _OBJECTDETECTIONARRAY._serialized_start=681
  _OBJECTDETECTIONARRAY._serialized_end=828
  _LANGUAGE._serialized_start=830
  _LANGUAGE._serialized_end=936
  _ROBOTPOSE._serialized_start=939
  _ROBOTPOSE._serialized_end=1152
  _OBJECTSFOUND._serialized_start=1154
  _OBJECTSFOUND._serialized_end=1256
# @@protoc_insertion_point(module_scope)
