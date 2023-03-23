# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: genmos_object_search/grpc/common.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n&genmos_object_search/grpc/common.proto\x12\x19genmos_object_search.grpc\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x19google/protobuf/any.proto\"V\n\x06Header\x12)\n\x05stamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x10\n\x08\x66rame_id\x18\x03 \x01(\t\x12\x0f\n\x07message\x18\x04 \x01(\t\"*\n\x06Pose2D\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\n\n\x02th\x18\x03 \x01(\x01\"t\n\x06Pose3D\x12\x31\n\x08position\x18\x01 \x01(\x0b\x32\x1f.genmos_object_search.grpc.Vec3\x12\x37\n\x08rotation\x18\x02 \x01(\x0b\x32%.genmos_object_search.grpc.Quaternion\"\'\n\x04Vec3\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"\x1c\n\x04Vec2\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\"8\n\nQuaternion\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\x12\t\n\x01w\x18\x04 \x01(\x01\"C\n\x05\x42ox2D\x12\r\n\x05x_min\x18\x01 \x01(\x01\x12\r\n\x05y_min\x18\x02 \x01(\x01\x12\r\n\x05x_max\x18\x03 \x01(\x01\x12\r\n\x05y_max\x18\x04 \x01(\x01\"j\n\x05\x42ox3D\x12\x31\n\x06\x63\x65nter\x18\x01 \x01(\x0b\x32!.genmos_object_search.grpc.Pose3D\x12.\n\x05sizes\x18\x02 \x01(\x0b\x32\x1f.genmos_object_search.grpc.Vec3\"Q\n\x07Voxel3D\x12,\n\x03pos\x18\x01 \x01(\x0b\x32\x1f.genmos_object_search.grpc.Vec3\x12\x10\n\x03res\x18\x03 \x01(\x01H\x00\x88\x01\x01\x42\x06\n\x04_res\"\x9a\x01\n\x11WeightedParticles\x12H\n\tparticles\x18\x01 \x03(\x0b\x32\x35.genmos_object_search.grpc.WeightedParticles.Particle\x1a;\n\x08Particle\x12\x1f\n\x01x\x18\x01 \x01(\x0b\x32\x14.google.protobuf.Any\x12\x0e\n\x06weight\x18\x02 \x01(\x01\"P\n\tHistogram\x12\x0e\n\x06length\x18\x01 \x01(\x03\x12$\n\x06values\x18\x02 \x03(\x0b\x32\x14.google.protobuf.Any\x12\r\n\x05probs\x18\x03 \x03(\x01*1\n\x06Status\x12\x0e\n\nSUCCESSFUL\x10\x00\x12\n\n\x06\x46\x41ILED\x10\x01\x12\x0b\n\x07PENDING\x10\x02\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'genmos_object_search.grpc.common_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _STATUS._serialized_start=1007
  _STATUS._serialized_end=1056
  _HEADER._serialized_start=129
  _HEADER._serialized_end=215
  _POSE2D._serialized_start=217
  _POSE2D._serialized_end=259
  _POSE3D._serialized_start=261
  _POSE3D._serialized_end=377
  _VEC3._serialized_start=379
  _VEC3._serialized_end=418
  _VEC2._serialized_start=420
  _VEC2._serialized_end=448
  _QUATERNION._serialized_start=450
  _QUATERNION._serialized_end=506
  _BOX2D._serialized_start=508
  _BOX2D._serialized_end=575
  _BOX3D._serialized_start=577
  _BOX3D._serialized_end=683
  _VOXEL3D._serialized_start=685
  _VOXEL3D._serialized_end=766
  _WEIGHTEDPARTICLES._serialized_start=769
  _WEIGHTEDPARTICLES._serialized_end=923
  _WEIGHTEDPARTICLES_PARTICLE._serialized_start=864
  _WEIGHTEDPARTICLES_PARTICLE._serialized_end=923
  _HISTOGRAM._serialized_start=925
  _HISTOGRAM._serialized_end=1005
# @@protoc_insertion_point(module_scope)
