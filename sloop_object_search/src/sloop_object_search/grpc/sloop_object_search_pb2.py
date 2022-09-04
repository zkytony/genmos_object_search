# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sloop_object_search/grpc/sloop_object_search.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from sloop_object_search.grpc import common_pb2 as sloop__object__search_dot_grpc_dot_common__pb2
from sloop_object_search.grpc import observation_pb2 as sloop__object__search_dot_grpc_dot_observation__pb2
from sloop_object_search.grpc import action_pb2 as sloop__object__search_dot_grpc_dot_action__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2sloop_object_search/grpc/sloop_object_search.proto\x12\x18sloop_object_search.grpc\x1a%sloop_object_search/grpc/common.proto\x1a*sloop_object_search/grpc/observation.proto\x1a%sloop_object_search/grpc/action.proto\"h\n\x12\x43reateAgentRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x03 \x01(\x0c\"\x87\x01\n\x10\x43reateAgentReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"c\n\x1dGetAgentCreationStatusRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\"\x99\x01\n\x1bGetAgentCreationStatusReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x16\n\x0estatus_message\x18\x03 \x01(\t\"\xda\x03\n\x19UpdateSearchRegionRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\r\n\x05is_3d\x18\x03 \x01(\x08\x12\x37\n\nrobot_pose\x18\x04 \x01(\x0b\x32#.sloop_object_search.grpc.RobotPose\x12\x41\n\x0eoccupancy_grid\x18\x05 \x01(\x0b\x32\'.sloop_object_search.grpc.OccupancyGridH\x00\x12;\n\x0bpoint_cloud\x18\x06 \x01(\x0b\x32$.sloop_object_search.grpc.PointCloudH\x00\x12O\n\x17search_region_params_2d\x18\x07 \x01(\x0b\x32..sloop_object_search.grpc.SearchRegionParams2D\x12O\n\x17search_region_params_3d\x18\x08 \x01(\x0b\x32..sloop_object_search.grpc.SearchRegionParams3DB\x0f\n\rsearch_region\"\x8e\x01\n\x17UpdateSearchRegionReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"\x89\x04\n\x19ProcessObservationRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x37\n\nrobot_pose\x18\x03 \x01(\x0b\x32#.sloop_object_search.grpc.RobotPose\x12=\n\robjects_found\x18\x06 \x01(\x0b\x32&.sloop_object_search.grpc.ObjectsFound\x12K\n\x11object_detections\x18\x04 \x01(\x0b\x32..sloop_object_search.grpc.ObjectDetectionArrayH\x00\x12\x36\n\x08language\x18\x05 \x01(\x0b\x32\".sloop_object_search.grpc.LanguageH\x00\x12\x16\n\taction_id\x18\x07 \x01(\tH\x01\x88\x01\x01\x12\x1c\n\x0f\x61\x63tion_finished\x18\x08 \x01(\x08H\x02\x88\x01\x01\x12\x17\n\nreturn_fov\x18\t \x01(\x08H\x03\x88\x01\x01\x12\x12\n\x05\x64\x65\x62ug\x18\n \x01(\x08H\x04\x88\x01\x01\x42\r\n\x0bobservationB\x0c\n\n_action_idB\x12\n\x10_action_finishedB\r\n\x0b_return_fovB\x08\n\x06_debug\"\x9c\x01\n\x17ProcessObservationReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12\x0c\n\x04\x66ovs\x18\x04 \x01(\x0c\"\xa1\x01\n\x17GetObjectBeliefsRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x12\n\nobject_ids\x18\x03 \x03(\t\x12.\n\x05scope\x18\x04 \x01(\x0e\x32\x1f.sloop_object_search.grpc.Scope\"\xcc\x01\n\x15GetObjectBeliefsReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12>\n\x0eobject_beliefs\x18\x04 \x03(\x0b\x32&.sloop_object_search.grpc.ObjectBelief\"\x8b\x01\n\x15GetRobotBeliefRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12.\n\x05scope\x18\x03 \x01(\x0e\x32\x1f.sloop_object_search.grpc.Scope\"\xc7\x01\n\x13GetRobotBeliefReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12;\n\x0crobot_belief\x18\x04 \x01(\x0b\x32%.sloop_object_search.grpc.RobotBelief\"}\n\x14\x43reatePlannerRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x04 \x01(\x0c\x12\x11\n\toverwrite\x18\x05 \x01(\x08\"\x89\x01\n\x12\x43reatePlannerReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"u\n\x11PlanActionRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x12\n\x05\x64\x65\x62ug\x18\x03 \x01(\x08H\x00\x88\x01\x01\x42\x08\n\x06_debug\"\xd9\x02\n\x0fPlanActionReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12>\n\x0bmove_action\x18\x04 \x01(\x0b\x32\'.sloop_object_search.grpc.MoveViewpointH\x00\x12\x35\n\x0b\x66ind_action\x18\x05 \x01(\x0b\x32\x1e.sloop_object_search.grpc.FindH\x00\x12=\n\tkv_action\x18\x06 \x01(\x0b\x32(.sloop_object_search.grpc.KeyValueActionH\x00\x12\x11\n\taction_id\x18\x07 \x01(\tB\x08\n\x06\x61\x63tion\"M\n\x08\x42idiNote\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x0f\n\x07message\x18\x02 \x01(\t\"\xfa\x01\n\x14SearchRegionParams2D\x12\x17\n\nlayout_cut\x18\x01 \x01(\x01H\x00\x88\x01\x01\x12\x16\n\tfloor_cut\x18\x02 \x01(\x01H\x01\x88\x01\x01\x12\x16\n\tgrid_size\x18\x03 \x01(\x01H\x02\x88\x01\x01\x12\x17\n\nbrush_size\x18\x04 \x01(\x01H\x03\x88\x01\x01\x12\x18\n\x0bregion_size\x18\x05 \x01(\x01H\x04\x88\x01\x01\x12\x12\n\x05\x64\x65\x62ug\x18\x06 \x01(\x08H\x05\x88\x01\x01\x42\r\n\x0b_layout_cutB\x0c\n\n_floor_cutB\x0c\n\n_grid_sizeB\r\n\x0b_brush_sizeB\x0e\n\x0c_region_sizeB\x08\n\x06_debug\"\xaa\x02\n\x14SearchRegionParams3D\x12\x18\n\x0boctree_size\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12$\n\x17search_space_resolution\x18\x02 \x01(\x01H\x01\x88\x01\x01\x12\x12\n\x05\x64\x65\x62ug\x18\x03 \x01(\x08H\x02\x88\x01\x01\x12\x1a\n\rregion_size_x\x18\x04 \x01(\x01H\x03\x88\x01\x01\x12\x1a\n\rregion_size_y\x18\x05 \x01(\x01H\x04\x88\x01\x01\x12\x1a\n\rregion_size_z\x18\x06 \x01(\x01H\x05\x88\x01\x01\x42\x0e\n\x0c_octree_sizeB\x1a\n\x18_search_space_resolutionB\x08\n\x06_debugB\x10\n\x0e_region_size_xB\x10\n\x0e_region_size_yB\x10\n\x0e_region_size_z\"\x84\x01\n\x0cObjectBelief\x12\x11\n\tobject_id\x18\x01 \x01(\t\x12\x31\n\x04\x64ist\x18\x02 \x01(\x0b\x32#.sloop_object_search.grpc.Histogram\x12.\n\x05scope\x18\x04 \x01(\x0e\x32\x1f.sloop_object_search.grpc.Scope\"\xf6\x01\n\x0bRobotBelief\x12\x10\n\x08robot_id\x18\x01 \x01(\t\x12\x31\n\x04pose\x18\x02 \x01(\x0b\x32#.sloop_object_search.grpc.RobotPose\x12=\n\robjects_found\x18\x03 \x01(\x0b\x32&.sloop_object_search.grpc.ObjectsFound\x12.\n\x05scope\x18\x04 \x01(\x0e\x32\x1f.sloop_object_search.grpc.Scope\x12\x33\n\x08topo_map\x18\x05 \x01(\x0b\x32!.sloop_object_search.grpc.TopoMap\"<\n\x07TopoMap\x12\x31\n\x05\x65\x64ges\x18\x01 \x03(\x0b\x32\".sloop_object_search.grpc.TopoEdge\"|\n\x08TopoEdge\x12\n\n\x02id\x18\x01 \x01(\t\x12\x31\n\x05node1\x18\x02 \x01(\x0b\x32\".sloop_object_search.grpc.TopoNode\x12\x31\n\x05node2\x18\x03 \x01(\x0b\x32\".sloop_object_search.grpc.TopoNode\"\x81\x01\n\x08TopoNode\x12\n\n\x02id\x18\x01 \x01(\t\x12\x30\n\x06pos_3d\x18\x02 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec3H\x00\x12\x30\n\x06pos_2d\x18\x03 \x01(\x0b\x32\x1e.sloop_object_search.grpc.Vec2H\x00\x42\x05\n\x03pos*\x1e\n\x05Scope\x12\t\n\x05LOCAL\x10\x00\x12\n\n\x06GLOBAL\x10\x01\x32\xd2\x07\n\x11SloopObjectSearch\x12i\n\x0b\x43reateAgent\x12,.sloop_object_search.grpc.CreateAgentRequest\x1a*.sloop_object_search.grpc.CreateAgentReply\"\x00\x12\x8a\x01\n\x16GetAgentCreationStatus\x12\x37.sloop_object_search.grpc.GetAgentCreationStatusRequest\x1a\x35.sloop_object_search.grpc.GetAgentCreationStatusReply\"\x00\x12~\n\x12UpdateSearchRegion\x12\x33.sloop_object_search.grpc.UpdateSearchRegionRequest\x1a\x31.sloop_object_search.grpc.UpdateSearchRegionReply\"\x00\x12~\n\x12ProcessObservation\x12\x33.sloop_object_search.grpc.ProcessObservationRequest\x1a\x31.sloop_object_search.grpc.ProcessObservationReply\"\x00\x12x\n\x10GetObjectBeliefs\x12\x31.sloop_object_search.grpc.GetObjectBeliefsRequest\x1a/.sloop_object_search.grpc.GetObjectBeliefsReply\"\x00\x12r\n\x0eGetRobotBelief\x12/.sloop_object_search.grpc.GetRobotBeliefRequest\x1a-.sloop_object_search.grpc.GetRobotBeliefReply\"\x00\x12o\n\rCreatePlanner\x12..sloop_object_search.grpc.CreatePlannerRequest\x1a,.sloop_object_search.grpc.CreatePlannerReply\"\x00\x12\x66\n\nPlanAction\x12+.sloop_object_search.grpc.PlanActionRequest\x1a).sloop_object_search.grpc.PlanActionReply\"\x00\x62\x06proto3')

_SCOPE = DESCRIPTOR.enum_types_by_name['Scope']
Scope = enum_type_wrapper.EnumTypeWrapper(_SCOPE)
LOCAL = 0
GLOBAL = 1


_CREATEAGENTREQUEST = DESCRIPTOR.message_types_by_name['CreateAgentRequest']
_CREATEAGENTREPLY = DESCRIPTOR.message_types_by_name['CreateAgentReply']
_GETAGENTCREATIONSTATUSREQUEST = DESCRIPTOR.message_types_by_name['GetAgentCreationStatusRequest']
_GETAGENTCREATIONSTATUSREPLY = DESCRIPTOR.message_types_by_name['GetAgentCreationStatusReply']
_UPDATESEARCHREGIONREQUEST = DESCRIPTOR.message_types_by_name['UpdateSearchRegionRequest']
_UPDATESEARCHREGIONREPLY = DESCRIPTOR.message_types_by_name['UpdateSearchRegionReply']
_PROCESSOBSERVATIONREQUEST = DESCRIPTOR.message_types_by_name['ProcessObservationRequest']
_PROCESSOBSERVATIONREPLY = DESCRIPTOR.message_types_by_name['ProcessObservationReply']
_GETOBJECTBELIEFSREQUEST = DESCRIPTOR.message_types_by_name['GetObjectBeliefsRequest']
_GETOBJECTBELIEFSREPLY = DESCRIPTOR.message_types_by_name['GetObjectBeliefsReply']
_GETROBOTBELIEFREQUEST = DESCRIPTOR.message_types_by_name['GetRobotBeliefRequest']
_GETROBOTBELIEFREPLY = DESCRIPTOR.message_types_by_name['GetRobotBeliefReply']
_CREATEPLANNERREQUEST = DESCRIPTOR.message_types_by_name['CreatePlannerRequest']
_CREATEPLANNERREPLY = DESCRIPTOR.message_types_by_name['CreatePlannerReply']
_PLANACTIONREQUEST = DESCRIPTOR.message_types_by_name['PlanActionRequest']
_PLANACTIONREPLY = DESCRIPTOR.message_types_by_name['PlanActionReply']
_BIDINOTE = DESCRIPTOR.message_types_by_name['BidiNote']
_SEARCHREGIONPARAMS2D = DESCRIPTOR.message_types_by_name['SearchRegionParams2D']
_SEARCHREGIONPARAMS3D = DESCRIPTOR.message_types_by_name['SearchRegionParams3D']
_OBJECTBELIEF = DESCRIPTOR.message_types_by_name['ObjectBelief']
_ROBOTBELIEF = DESCRIPTOR.message_types_by_name['RobotBelief']
_TOPOMAP = DESCRIPTOR.message_types_by_name['TopoMap']
_TOPOEDGE = DESCRIPTOR.message_types_by_name['TopoEdge']
_TOPONODE = DESCRIPTOR.message_types_by_name['TopoNode']
CreateAgentRequest = _reflection.GeneratedProtocolMessageType('CreateAgentRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATEAGENTREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.CreateAgentRequest)
  })
_sym_db.RegisterMessage(CreateAgentRequest)

CreateAgentReply = _reflection.GeneratedProtocolMessageType('CreateAgentReply', (_message.Message,), {
  'DESCRIPTOR' : _CREATEAGENTREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.CreateAgentReply)
  })
_sym_db.RegisterMessage(CreateAgentReply)

GetAgentCreationStatusRequest = _reflection.GeneratedProtocolMessageType('GetAgentCreationStatusRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETAGENTCREATIONSTATUSREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.GetAgentCreationStatusRequest)
  })
_sym_db.RegisterMessage(GetAgentCreationStatusRequest)

GetAgentCreationStatusReply = _reflection.GeneratedProtocolMessageType('GetAgentCreationStatusReply', (_message.Message,), {
  'DESCRIPTOR' : _GETAGENTCREATIONSTATUSREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.GetAgentCreationStatusReply)
  })
_sym_db.RegisterMessage(GetAgentCreationStatusReply)

UpdateSearchRegionRequest = _reflection.GeneratedProtocolMessageType('UpdateSearchRegionRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATESEARCHREGIONREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.UpdateSearchRegionRequest)
  })
_sym_db.RegisterMessage(UpdateSearchRegionRequest)

UpdateSearchRegionReply = _reflection.GeneratedProtocolMessageType('UpdateSearchRegionReply', (_message.Message,), {
  'DESCRIPTOR' : _UPDATESEARCHREGIONREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.UpdateSearchRegionReply)
  })
_sym_db.RegisterMessage(UpdateSearchRegionReply)

ProcessObservationRequest = _reflection.GeneratedProtocolMessageType('ProcessObservationRequest', (_message.Message,), {
  'DESCRIPTOR' : _PROCESSOBSERVATIONREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ProcessObservationRequest)
  })
_sym_db.RegisterMessage(ProcessObservationRequest)

ProcessObservationReply = _reflection.GeneratedProtocolMessageType('ProcessObservationReply', (_message.Message,), {
  'DESCRIPTOR' : _PROCESSOBSERVATIONREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ProcessObservationReply)
  })
_sym_db.RegisterMessage(ProcessObservationReply)

GetObjectBeliefsRequest = _reflection.GeneratedProtocolMessageType('GetObjectBeliefsRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETOBJECTBELIEFSREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.GetObjectBeliefsRequest)
  })
_sym_db.RegisterMessage(GetObjectBeliefsRequest)

GetObjectBeliefsReply = _reflection.GeneratedProtocolMessageType('GetObjectBeliefsReply', (_message.Message,), {
  'DESCRIPTOR' : _GETOBJECTBELIEFSREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.GetObjectBeliefsReply)
  })
_sym_db.RegisterMessage(GetObjectBeliefsReply)

GetRobotBeliefRequest = _reflection.GeneratedProtocolMessageType('GetRobotBeliefRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETROBOTBELIEFREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.GetRobotBeliefRequest)
  })
_sym_db.RegisterMessage(GetRobotBeliefRequest)

GetRobotBeliefReply = _reflection.GeneratedProtocolMessageType('GetRobotBeliefReply', (_message.Message,), {
  'DESCRIPTOR' : _GETROBOTBELIEFREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.GetRobotBeliefReply)
  })
_sym_db.RegisterMessage(GetRobotBeliefReply)

CreatePlannerRequest = _reflection.GeneratedProtocolMessageType('CreatePlannerRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATEPLANNERREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.CreatePlannerRequest)
  })
_sym_db.RegisterMessage(CreatePlannerRequest)

CreatePlannerReply = _reflection.GeneratedProtocolMessageType('CreatePlannerReply', (_message.Message,), {
  'DESCRIPTOR' : _CREATEPLANNERREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.CreatePlannerReply)
  })
_sym_db.RegisterMessage(CreatePlannerReply)

PlanActionRequest = _reflection.GeneratedProtocolMessageType('PlanActionRequest', (_message.Message,), {
  'DESCRIPTOR' : _PLANACTIONREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.PlanActionRequest)
  })
_sym_db.RegisterMessage(PlanActionRequest)

PlanActionReply = _reflection.GeneratedProtocolMessageType('PlanActionReply', (_message.Message,), {
  'DESCRIPTOR' : _PLANACTIONREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.PlanActionReply)
  })
_sym_db.RegisterMessage(PlanActionReply)

BidiNote = _reflection.GeneratedProtocolMessageType('BidiNote', (_message.Message,), {
  'DESCRIPTOR' : _BIDINOTE,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.BidiNote)
  })
_sym_db.RegisterMessage(BidiNote)

SearchRegionParams2D = _reflection.GeneratedProtocolMessageType('SearchRegionParams2D', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREGIONPARAMS2D,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.SearchRegionParams2D)
  })
_sym_db.RegisterMessage(SearchRegionParams2D)

SearchRegionParams3D = _reflection.GeneratedProtocolMessageType('SearchRegionParams3D', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREGIONPARAMS3D,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.SearchRegionParams3D)
  })
_sym_db.RegisterMessage(SearchRegionParams3D)

ObjectBelief = _reflection.GeneratedProtocolMessageType('ObjectBelief', (_message.Message,), {
  'DESCRIPTOR' : _OBJECTBELIEF,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ObjectBelief)
  })
_sym_db.RegisterMessage(ObjectBelief)

RobotBelief = _reflection.GeneratedProtocolMessageType('RobotBelief', (_message.Message,), {
  'DESCRIPTOR' : _ROBOTBELIEF,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.RobotBelief)
  })
_sym_db.RegisterMessage(RobotBelief)

TopoMap = _reflection.GeneratedProtocolMessageType('TopoMap', (_message.Message,), {
  'DESCRIPTOR' : _TOPOMAP,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.TopoMap)
  })
_sym_db.RegisterMessage(TopoMap)

TopoEdge = _reflection.GeneratedProtocolMessageType('TopoEdge', (_message.Message,), {
  'DESCRIPTOR' : _TOPOEDGE,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.TopoEdge)
  })
_sym_db.RegisterMessage(TopoEdge)

TopoNode = _reflection.GeneratedProtocolMessageType('TopoNode', (_message.Message,), {
  'DESCRIPTOR' : _TOPONODE,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.TopoNode)
  })
_sym_db.RegisterMessage(TopoNode)

_SLOOPOBJECTSEARCH = DESCRIPTOR.services_by_name['SloopObjectSearch']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SCOPE._serialized_start=4794
  _SCOPE._serialized_end=4824
  _CREATEAGENTREQUEST._serialized_start=202
  _CREATEAGENTREQUEST._serialized_end=306
  _CREATEAGENTREPLY._serialized_start=309
  _CREATEAGENTREPLY._serialized_end=444
  _GETAGENTCREATIONSTATUSREQUEST._serialized_start=446
  _GETAGENTCREATIONSTATUSREQUEST._serialized_end=545
  _GETAGENTCREATIONSTATUSREPLY._serialized_start=548
  _GETAGENTCREATIONSTATUSREPLY._serialized_end=701
  _UPDATESEARCHREGIONREQUEST._serialized_start=704
  _UPDATESEARCHREGIONREQUEST._serialized_end=1178
  _UPDATESEARCHREGIONREPLY._serialized_start=1181
  _UPDATESEARCHREGIONREPLY._serialized_end=1323
  _PROCESSOBSERVATIONREQUEST._serialized_start=1326
  _PROCESSOBSERVATIONREQUEST._serialized_end=1847
  _PROCESSOBSERVATIONREPLY._serialized_start=1850
  _PROCESSOBSERVATIONREPLY._serialized_end=2006
  _GETOBJECTBELIEFSREQUEST._serialized_start=2009
  _GETOBJECTBELIEFSREQUEST._serialized_end=2170
  _GETOBJECTBELIEFSREPLY._serialized_start=2173
  _GETOBJECTBELIEFSREPLY._serialized_end=2377
  _GETROBOTBELIEFREQUEST._serialized_start=2380
  _GETROBOTBELIEFREQUEST._serialized_end=2519
  _GETROBOTBELIEFREPLY._serialized_start=2522
  _GETROBOTBELIEFREPLY._serialized_end=2721
  _CREATEPLANNERREQUEST._serialized_start=2723
  _CREATEPLANNERREQUEST._serialized_end=2848
  _CREATEPLANNERREPLY._serialized_start=2851
  _CREATEPLANNERREPLY._serialized_end=2988
  _PLANACTIONREQUEST._serialized_start=2990
  _PLANACTIONREQUEST._serialized_end=3107
  _PLANACTIONREPLY._serialized_start=3110
  _PLANACTIONREPLY._serialized_end=3455
  _BIDINOTE._serialized_start=3457
  _BIDINOTE._serialized_end=3534
  _SEARCHREGIONPARAMS2D._serialized_start=3537
  _SEARCHREGIONPARAMS2D._serialized_end=3787
  _SEARCHREGIONPARAMS3D._serialized_start=3790
  _SEARCHREGIONPARAMS3D._serialized_end=4088
  _OBJECTBELIEF._serialized_start=4091
  _OBJECTBELIEF._serialized_end=4223
  _ROBOTBELIEF._serialized_start=4226
  _ROBOTBELIEF._serialized_end=4472
  _TOPOMAP._serialized_start=4474
  _TOPOMAP._serialized_end=4534
  _TOPOEDGE._serialized_start=4536
  _TOPOEDGE._serialized_end=4660
  _TOPONODE._serialized_start=4663
  _TOPONODE._serialized_end=4792
  _SLOOPOBJECTSEARCH._serialized_start=4827
  _SLOOPOBJECTSEARCH._serialized_end=5805
# @@protoc_insertion_point(module_scope)
