# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: genmos_object_search/grpc/action.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from genmos_object_search.grpc import common_pb2 as genmos__object__search_dot_grpc_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n&genmos_object_search/grpc/action.proto\x12\x19genmos_object_search.grpc\x1a&genmos_object_search/grpc/common.proto\"\xb4\x01\n\x08Motion3D\x12-\n\x04\x64pos\x18\x01 \x01(\x0b\x32\x1f.genmos_object_search.grpc.Vec3\x12\x35\n\ndrot_euler\x18\x02 \x01(\x0b\x32\x1f.genmos_object_search.grpc.Vec3H\x00\x12:\n\tdrot_quat\x18\x03 \x01(\x0b\x32%.genmos_object_search.grpc.QuaternionH\x00\x42\x06\n\x04\x64rot\"e\n\x08Motion2D\x12/\n\x04\x64pos\x18\x01 \x01(\x0b\x32\x1f.genmos_object_search.grpc.Vec2H\x00\x12\x11\n\x07\x66orward\x18\x02 \x01(\x01H\x00\x12\x0b\n\x03\x64th\x18\x03 \x01(\x01\x42\x08\n\x06\x64trans\"\xeb\x02\n\rMoveViewpoint\x12\x31\n\x06header\x18\x01 \x01(\x0b\x32!.genmos_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x34\n\x07\x64\x65st_3d\x18\x03 \x01(\x0b\x32!.genmos_object_search.grpc.Pose3DH\x00\x12\x34\n\x07\x64\x65st_2d\x18\x04 \x01(\x0b\x32!.genmos_object_search.grpc.Pose2DH\x00\x12\x38\n\tmotion_2d\x18\x05 \x01(\x0b\x32#.genmos_object_search.grpc.Motion2DH\x01\x12\x38\n\tmotion_3d\x18\x06 \x01(\x0b\x32#.genmos_object_search.grpc.Motion3DH\x01\x12\x0c\n\x04name\x18\x07 \x01(\t\x12\x15\n\rexpected_cost\x18\x08 \x01(\x02\x42\x06\n\x04poseB\x08\n\x06motion\"Y\n\x04\x46ind\x12\x31\n\x06header\x18\x01 \x01(\x0b\x32!.genmos_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\"\x95\x01\n\x0eKeyValueAction\x12\x31\n\x06header\x18\x01 \x01(\x0b\x32!.genmos_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x12\n\nnum_fields\x18\x04 \x01(\x05\x12\x0c\n\x04keys\x18\x05 \x03(\t\x12\x0e\n\x06values\x18\x06 \x03(\tb\x06proto3')



_MOTION3D = DESCRIPTOR.message_types_by_name['Motion3D']
_MOTION2D = DESCRIPTOR.message_types_by_name['Motion2D']
_MOVEVIEWPOINT = DESCRIPTOR.message_types_by_name['MoveViewpoint']
_FIND = DESCRIPTOR.message_types_by_name['Find']
_KEYVALUEACTION = DESCRIPTOR.message_types_by_name['KeyValueAction']
Motion3D = _reflection.GeneratedProtocolMessageType('Motion3D', (_message.Message,), {
  'DESCRIPTOR' : _MOTION3D,
  '__module__' : 'genmos_object_search.grpc.action_pb2'
  # @@protoc_insertion_point(class_scope:genmos_object_search.grpc.Motion3D)
  })
_sym_db.RegisterMessage(Motion3D)

Motion2D = _reflection.GeneratedProtocolMessageType('Motion2D', (_message.Message,), {
  'DESCRIPTOR' : _MOTION2D,
  '__module__' : 'genmos_object_search.grpc.action_pb2'
  # @@protoc_insertion_point(class_scope:genmos_object_search.grpc.Motion2D)
  })
_sym_db.RegisterMessage(Motion2D)

MoveViewpoint = _reflection.GeneratedProtocolMessageType('MoveViewpoint', (_message.Message,), {
  'DESCRIPTOR' : _MOVEVIEWPOINT,
  '__module__' : 'genmos_object_search.grpc.action_pb2'
  # @@protoc_insertion_point(class_scope:genmos_object_search.grpc.MoveViewpoint)
  })
_sym_db.RegisterMessage(MoveViewpoint)

Find = _reflection.GeneratedProtocolMessageType('Find', (_message.Message,), {
  'DESCRIPTOR' : _FIND,
  '__module__' : 'genmos_object_search.grpc.action_pb2'
  # @@protoc_insertion_point(class_scope:genmos_object_search.grpc.Find)
  })
_sym_db.RegisterMessage(Find)

KeyValueAction = _reflection.GeneratedProtocolMessageType('KeyValueAction', (_message.Message,), {
  'DESCRIPTOR' : _KEYVALUEACTION,
  '__module__' : 'genmos_object_search.grpc.action_pb2'
  # @@protoc_insertion_point(class_scope:genmos_object_search.grpc.KeyValueAction)
  })
_sym_db.RegisterMessage(KeyValueAction)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MOTION3D._serialized_start=110
  _MOTION3D._serialized_end=290
  _MOTION2D._serialized_start=292
  _MOTION2D._serialized_end=393
  _MOVEVIEWPOINT._serialized_start=396
  _MOVEVIEWPOINT._serialized_end=759
  _FIND._serialized_start=761
  _FIND._serialized_end=850
  _KEYVALUEACTION._serialized_start=853
  _KEYVALUEACTION._serialized_end=1002
# @@protoc_insertion_point(module_scope)
