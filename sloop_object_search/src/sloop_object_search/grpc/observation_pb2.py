# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sloop_object_search/grpc/observation.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from sloop_object_search.grpc import common_pb2 as sloop__object__search_dot_grpc_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*sloop_object_search/grpc/observation.proto\x12\x18sloop_object_search.grpc\x1a%sloop_object_search/grpc/common.proto\"\xe2\x01\n\rOccupancyGrid\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12;\n\x05grids\x18\x03 \x03(\x0b\x32,.sloop_object_search.grpc.OccupancyGrid.Grid\x1aP\n\x04Grid\x12+\n\x03pos\x18\x01 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec2\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\r\n\x05label\x18\x03 \x01(\t\"\xd1\x01\n\nPointCloud\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12:\n\x06points\x18\x03 \x03(\x0b\x32*.sloop_object_search.grpc.PointCloud.Point\x1a\x43\n\x05Point\x12+\n\x03pos\x18\x01 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec3\x12\r\n\x05label\x18\x02 \x01(\t\"\x80\x02\n\x0fObjectDetection\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12I\n\ndetections\x18\x03 \x03(\x0b\x32\x35.sloop_object_search.grpc.ObjectDetection.Detection3D\x1a^\n\x0b\x44\x65tection3D\x12\r\n\x05label\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12,\n\x03\x62ox\x18\x03 \x01(\x0b\x32\x1f.sloop_object_search.grpc.Box3D\"\\\n\x08Language\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0c\n\x04text\x18\x03 \x01(\tb\x06proto3')



_OCCUPANCYGRID = DESCRIPTOR.message_types_by_name['OccupancyGrid']
_OCCUPANCYGRID_GRID = _OCCUPANCYGRID.nested_types_by_name['Grid']
_POINTCLOUD = DESCRIPTOR.message_types_by_name['PointCloud']
_POINTCLOUD_POINT = _POINTCLOUD.nested_types_by_name['Point']
_OBJECTDETECTION = DESCRIPTOR.message_types_by_name['ObjectDetection']
_OBJECTDETECTION_DETECTION3D = _OBJECTDETECTION.nested_types_by_name['Detection3D']
_LANGUAGE = DESCRIPTOR.message_types_by_name['Language']
OccupancyGrid = _reflection.GeneratedProtocolMessageType('OccupancyGrid', (_message.Message,), {

  'Grid' : _reflection.GeneratedProtocolMessageType('Grid', (_message.Message,), {
    'DESCRIPTOR' : _OCCUPANCYGRID_GRID,
    '__module__' : 'sloop_object_search.grpc.observation_pb2'
    # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.OccupancyGrid.Grid)
    })
  ,
  'DESCRIPTOR' : _OCCUPANCYGRID,
  '__module__' : 'sloop_object_search.grpc.observation_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.OccupancyGrid)
  })
_sym_db.RegisterMessage(OccupancyGrid)
_sym_db.RegisterMessage(OccupancyGrid.Grid)

PointCloud = _reflection.GeneratedProtocolMessageType('PointCloud', (_message.Message,), {

  'Point' : _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), {
    'DESCRIPTOR' : _POINTCLOUD_POINT,
    '__module__' : 'sloop_object_search.grpc.observation_pb2'
    # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.PointCloud.Point)
    })
  ,
  'DESCRIPTOR' : _POINTCLOUD,
  '__module__' : 'sloop_object_search.grpc.observation_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.PointCloud)
  })
_sym_db.RegisterMessage(PointCloud)
_sym_db.RegisterMessage(PointCloud.Point)

ObjectDetection = _reflection.GeneratedProtocolMessageType('ObjectDetection', (_message.Message,), {

  'Detection3D' : _reflection.GeneratedProtocolMessageType('Detection3D', (_message.Message,), {
    'DESCRIPTOR' : _OBJECTDETECTION_DETECTION3D,
    '__module__' : 'sloop_object_search.grpc.observation_pb2'
    # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ObjectDetection.Detection3D)
    })
  ,
  'DESCRIPTOR' : _OBJECTDETECTION,
  '__module__' : 'sloop_object_search.grpc.observation_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ObjectDetection)
  })
_sym_db.RegisterMessage(ObjectDetection)
_sym_db.RegisterMessage(ObjectDetection.Detection3D)

Language = _reflection.GeneratedProtocolMessageType('Language', (_message.Message,), {
  'DESCRIPTOR' : _LANGUAGE,
  '__module__' : 'sloop_object_search.grpc.observation_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.Language)
  })
_sym_db.RegisterMessage(Language)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OCCUPANCYGRID._serialized_start=112
  _OCCUPANCYGRID._serialized_end=338
  _OCCUPANCYGRID_GRID._serialized_start=258
  _OCCUPANCYGRID_GRID._serialized_end=338
  _POINTCLOUD._serialized_start=341
  _POINTCLOUD._serialized_end=550
  _POINTCLOUD_POINT._serialized_start=483
  _POINTCLOUD_POINT._serialized_end=550
  _OBJECTDETECTION._serialized_start=553
  _OBJECTDETECTION._serialized_end=809
  _OBJECTDETECTION_DETECTION3D._serialized_start=715
  _OBJECTDETECTION_DETECTION3D._serialized_end=809
  _LANGUAGE._serialized_start=811
  _LANGUAGE._serialized_end=903
# @@protoc_insertion_point(module_scope)