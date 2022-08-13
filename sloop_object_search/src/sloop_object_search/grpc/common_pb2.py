# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sloop_object_search/grpc/common.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n%sloop_object_search/grpc/common.proto\x12\x18sloop_object_search.grpc\x1a\x1fgoogle/protobuf/timestamp.proto\"V\n\x06Header\x12)\n\x05stamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x10\n\x08\x66rame_id\x18\x03 \x01(\t\x12\x0f\n\x07message\x18\x04 \x01(\t\"*\n\x06Pose2D\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\n\n\x02th\x18\x03 \x01(\x01\"r\n\x06Pose3D\x12\x30\n\x08position\x18\x01 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec3\x12\x36\n\x08rotation\x18\x02 \x01(\x0b\x32$.sloop_object_search.grpc.Quaternion\"\'\n\x04Vec3\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"\x1c\n\x04Vec2\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\"8\n\nQuaternion\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\x12\t\n\x01w\x18\x04 \x01(\x01\"g\n\x05\x42ox3D\x12\x30\n\x06\x63\x65nter\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Pose3D\x12,\n\x04size\x18\x02 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec3*!\n\x06Status\x12\x0b\n\x07SUCCESS\x10\x00\x12\n\n\x06\x46\x41ILED\x10\x01\x62\x06proto3')

_STATUS = DESCRIPTOR.enum_types_by_name['Status']
Status = enum_type_wrapper.EnumTypeWrapper(_STATUS)
SUCCESS = 0
FAILED = 1


_HEADER = DESCRIPTOR.message_types_by_name['Header']
_POSE2D = DESCRIPTOR.message_types_by_name['Pose2D']
_POSE3D = DESCRIPTOR.message_types_by_name['Pose3D']
_VEC3 = DESCRIPTOR.message_types_by_name['Vec3']
_VEC2 = DESCRIPTOR.message_types_by_name['Vec2']
_QUATERNION = DESCRIPTOR.message_types_by_name['Quaternion']
_BOX3D = DESCRIPTOR.message_types_by_name['Box3D']
Header = _reflection.GeneratedProtocolMessageType('Header', (_message.Message,), {
  'DESCRIPTOR' : _HEADER,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Header)
  })
_sym_db.RegisterMessage(Header)

Pose2D = _reflection.GeneratedProtocolMessageType('Pose2D', (_message.Message,), {
  'DESCRIPTOR' : _POSE2D,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Pose2D)
  })
_sym_db.RegisterMessage(Pose2D)

Pose3D = _reflection.GeneratedProtocolMessageType('Pose3D', (_message.Message,), {
  'DESCRIPTOR' : _POSE3D,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Pose3D)
  })
_sym_db.RegisterMessage(Pose3D)

Vec3 = _reflection.GeneratedProtocolMessageType('Vec3', (_message.Message,), {
  'DESCRIPTOR' : _VEC3,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Vec3)
  })
_sym_db.RegisterMessage(Vec3)

Vec2 = _reflection.GeneratedProtocolMessageType('Vec2', (_message.Message,), {
  'DESCRIPTOR' : _VEC2,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Vec2)
  })
_sym_db.RegisterMessage(Vec2)

Quaternion = _reflection.GeneratedProtocolMessageType('Quaternion', (_message.Message,), {
  'DESCRIPTOR' : _QUATERNION,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Quaternion)
  })
_sym_db.RegisterMessage(Quaternion)

Box3D = _reflection.GeneratedProtocolMessageType('Box3D', (_message.Message,), {
  'DESCRIPTOR' : _BOX3D,
  '__module__' : 'sloop_object_search.grpc.common_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Box3D)
  })
_sym_db.RegisterMessage(Box3D)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _STATUS._serialized_start=582
  _STATUS._serialized_end=615
  _HEADER._serialized_start=100
  _HEADER._serialized_end=186
  _POSE2D._serialized_start=188
  _POSE2D._serialized_end=230
  _POSE3D._serialized_start=232
  _POSE3D._serialized_end=346
  _VEC3._serialized_start=348
  _VEC3._serialized_end=387
  _VEC2._serialized_start=389
  _VEC2._serialized_end=417
  _QUATERNION._serialized_start=419
  _QUATERNION._serialized_end=475
  _BOX3D._serialized_start=477
  _BOX3D._serialized_end=580
# @@protoc_insertion_point(module_scope)
