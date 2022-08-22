# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sloop_object_search/grpc/sloop_object_search.proto
"""Generated protocol buffer code."""
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2sloop_object_search/grpc/sloop_object_search.proto\x12\x18sloop_object_search.grpc\x1a%sloop_object_search/grpc/common.proto\x1a*sloop_object_search/grpc/observation.proto\x1a%sloop_object_search/grpc/action.proto\"h\n\x12\x43reateAgentRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x03 \x01(\x0c\"\x87\x01\n\x10\x43reateAgentReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"c\n\x1dGetAgentCreationStatusRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\"\x99\x01\n\x1bGetAgentCreationStatusReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x16\n\x0estatus_message\x18\x03 \x01(\t\"\xf6\x03\n\x19UpdateSearchRegionRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\r\n\x05is_3d\x18\x03 \x01(\x08\x12\x37\n\nrobot_pose\x18\x04 \x01(\x0b\x32#.sloop_object_search.grpc.RobotPose\x12\x41\n\x0eoccupancy_grid\x18\x05 \x01(\x0b\x32\'.sloop_object_search.grpc.OccupancyGridH\x00\x12;\n\x0bpoint_cloud\x18\x06 \x01(\x0b\x32$.sloop_object_search.grpc.PointCloudH\x00\x12Q\n\x17search_region_params_2d\x18\x07 \x01(\x0b\x32..sloop_object_search.grpc.SearchRegionParams2DH\x01\x12Q\n\x17search_region_params_3d\x18\x08 \x01(\x0b\x32..sloop_object_search.grpc.SearchRegionParams3DH\x01\x42\x0f\n\rsearch_regionB\x16\n\x14search_region_params\"\x8e\x01\n\x17UpdateSearchRegionReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"\xd4\x02\n\x19ProcessObservationRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x39\n\nrobot_pose\x18\x03 \x01(\x0b\x32#.sloop_object_search.grpc.RobotPoseH\x00\x12K\n\x11object_detections\x18\x04 \x01(\x0b\x32..sloop_object_search.grpc.ObjectDetectionArrayH\x00\x12\x36\n\x08language\x18\x05 \x01(\x0b\x32\".sloop_object_search.grpc.LanguageH\x00\x12\x16\n\taction_id\x18\x06 \x01(\tH\x01\x88\x01\x01\x42\r\n\x0bobservationB\x0c\n\n_action_id\"\x8e\x01\n\x17ProcessObservationReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"q\n\x17GetObjectBeliefsRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x12\n\nobject_ids\x18\x03 \x03(\t\"\xcc\x01\n\x15GetObjectBeliefsReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12>\n\x0eobject_beliefs\x18\x04 \x03(\x0b\x32&.sloop_object_search.grpc.ObjectBelief\"[\n\x15GetRobotBeliefRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\"\xc7\x01\n\x13GetRobotBeliefReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12;\n\x0crobot_belief\x18\x04 \x01(\x0b\x32%.sloop_object_search.grpc.RobotBelief\"}\n\x14\x43reatePlannerRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x0e\n\x06\x63onfig\x18\x04 \x01(\x0c\x12\x11\n\toverwrite\x18\x05 \x01(\x08\"\x89\x01\n\x12\x43reatePlannerReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"W\n\x11PlanActionRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\"\xd9\x02\n\x0fPlanActionReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\x12>\n\x0bmove_action\x18\x04 \x01(\x0b\x32\'.sloop_object_search.grpc.MoveViewpointH\x00\x12\x35\n\x0b\x66ind_action\x18\x05 \x01(\x0b\x32\x1e.sloop_object_search.grpc.FindH\x00\x12=\n\tkv_action\x18\x06 \x01(\x0b\x32(.sloop_object_search.grpc.KeyValueActionH\x00\x12\x11\n\taction_id\x18\x07 \x01(\tB\x08\n\x06\x61\x63tion\"n\n\x15\x41\x63tionFinishedRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x11\n\taction_id\x18\x03 \x01(\t\"\x8a\x01\n\x13\x41\x63tionFinishedReply\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .sloop_object_search.grpc.Header\x12\x30\n\x06status\x18\x02 \x01(\x0e\x32 .sloop_object_search.grpc.Status\x12\x0f\n\x07message\x18\x03 \x01(\t\"\xfa\x01\n\x14SearchRegionParams2D\x12\x17\n\nlayout_cut\x18\x01 \x01(\x01H\x00\x88\x01\x01\x12\x16\n\tfloor_cut\x18\x02 \x01(\x01H\x01\x88\x01\x01\x12\x16\n\tgrid_size\x18\x03 \x01(\x01H\x02\x88\x01\x01\x12\x17\n\nbrush_size\x18\x04 \x01(\x01H\x03\x88\x01\x01\x12\x18\n\x0bregion_size\x18\x05 \x01(\x01H\x04\x88\x01\x01\x12\x12\n\x05\x64\x65\x62ug\x18\x06 \x01(\x08H\x05\x88\x01\x01\x42\r\n\x0b_layout_cutB\x0c\n\n_floor_cutB\x0c\n\n_grid_sizeB\r\n\x0b_brush_sizeB\x0e\n\x0c_region_sizeB\x08\n\x06_debug\"\xaa\x02\n\x14SearchRegionParams3D\x12\x18\n\x0boctree_size\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12$\n\x17search_space_resolution\x18\x02 \x01(\x01H\x01\x88\x01\x01\x12\x12\n\x05\x64\x65\x62ug\x18\x03 \x01(\x08H\x02\x88\x01\x01\x12\x1a\n\rregion_size_x\x18\x04 \x01(\x01H\x03\x88\x01\x01\x12\x1a\n\rregion_size_y\x18\x05 \x01(\x01H\x04\x88\x01\x01\x12\x1a\n\rregion_size_z\x18\x06 \x01(\x01H\x05\x88\x01\x01\x42\x0e\n\x0c_octree_sizeB\x1a\n\x18_search_space_resolutionB\x08\n\x06_debugB\x10\n\x0e_region_size_xB\x10\n\x0e_region_size_yB\x10\n\x0e_region_size_z\"T\n\x0cObjectBelief\x12\x11\n\tobject_id\x18\x02 \x01(\t\x12\x31\n\x04\x64ist\x18\x03 \x01(\x0b\x32#.sloop_object_search.grpc.Histogram\"\x8d\x01\n\x0bRobotBelief\x12\x10\n\x08robot_id\x18\x01 \x01(\t\x12\x15\n\robjects_found\x18\x02 \x03(\t\x12\x31\n\x04pose\x18\x03 \x01(\x0b\x32#.sloop_object_search.grpc.RobotPose\x12\x15\n\x08topo_nid\x18\x04 \x01(\tH\x00\x88\x01\x01\x42\x0b\n\t_topo_nid2\xc6\x08\n\x11SloopObjectSearch\x12i\n\x0b\x43reateAgent\x12,.sloop_object_search.grpc.CreateAgentRequest\x1a*.sloop_object_search.grpc.CreateAgentReply\"\x00\x12\x8a\x01\n\x16GetAgentCreationStatus\x12\x37.sloop_object_search.grpc.GetAgentCreationStatusRequest\x1a\x35.sloop_object_search.grpc.GetAgentCreationStatusReply\"\x00\x12~\n\x12UpdateSearchRegion\x12\x33.sloop_object_search.grpc.UpdateSearchRegionRequest\x1a\x31.sloop_object_search.grpc.UpdateSearchRegionReply\"\x00\x12~\n\x12ProcessObservation\x12\x33.sloop_object_search.grpc.ProcessObservationRequest\x1a\x31.sloop_object_search.grpc.ProcessObservationReply\"\x00\x12x\n\x10GetObjectBeliefs\x12\x31.sloop_object_search.grpc.GetObjectBeliefsRequest\x1a/.sloop_object_search.grpc.GetObjectBeliefsReply\"\x00\x12r\n\x0eGetRobotBelief\x12/.sloop_object_search.grpc.GetRobotBeliefRequest\x1a-.sloop_object_search.grpc.GetRobotBeliefReply\"\x00\x12o\n\rCreatePlanner\x12..sloop_object_search.grpc.CreatePlannerRequest\x1a,.sloop_object_search.grpc.CreatePlannerReply\"\x00\x12\x66\n\nPlanAction\x12+.sloop_object_search.grpc.PlanActionRequest\x1a).sloop_object_search.grpc.PlanActionReply\"\x00\x12r\n\x0e\x41\x63tionFinished\x12/.sloop_object_search.grpc.ActionFinishedRequest\x1a-.sloop_object_search.grpc.ActionFinishedReply\"\x00\x62\x06proto3')



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
_ACTIONFINISHEDREQUEST = DESCRIPTOR.message_types_by_name['ActionFinishedRequest']
_ACTIONFINISHEDREPLY = DESCRIPTOR.message_types_by_name['ActionFinishedReply']
_SEARCHREGIONPARAMS2D = DESCRIPTOR.message_types_by_name['SearchRegionParams2D']
_SEARCHREGIONPARAMS3D = DESCRIPTOR.message_types_by_name['SearchRegionParams3D']
_OBJECTBELIEF = DESCRIPTOR.message_types_by_name['ObjectBelief']
_ROBOTBELIEF = DESCRIPTOR.message_types_by_name['RobotBelief']
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

ActionFinishedRequest = _reflection.GeneratedProtocolMessageType('ActionFinishedRequest', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONFINISHEDREQUEST,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ActionFinishedRequest)
  })
_sym_db.RegisterMessage(ActionFinishedRequest)

ActionFinishedReply = _reflection.GeneratedProtocolMessageType('ActionFinishedReply', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONFINISHEDREPLY,
  '__module__' : 'sloop_object_search.grpc.sloop_object_search_pb2'
  # @@protoc_insertion_point(class_scope:sloop_object_search.grpc.ActionFinishedReply)
  })
_sym_db.RegisterMessage(ActionFinishedReply)

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

_SLOOPOBJECTSEARCH = DESCRIPTOR.services_by_name['SloopObjectSearch']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CREATEAGENTREQUEST._serialized_start=202
  _CREATEAGENTREQUEST._serialized_end=306
  _CREATEAGENTREPLY._serialized_start=309
  _CREATEAGENTREPLY._serialized_end=444
  _GETAGENTCREATIONSTATUSREQUEST._serialized_start=446
  _GETAGENTCREATIONSTATUSREQUEST._serialized_end=545
  _GETAGENTCREATIONSTATUSREPLY._serialized_start=548
  _GETAGENTCREATIONSTATUSREPLY._serialized_end=701
  _UPDATESEARCHREGIONREQUEST._serialized_start=704
  _UPDATESEARCHREGIONREQUEST._serialized_end=1206
  _UPDATESEARCHREGIONREPLY._serialized_start=1209
  _UPDATESEARCHREGIONREPLY._serialized_end=1351
  _PROCESSOBSERVATIONREQUEST._serialized_start=1354
  _PROCESSOBSERVATIONREQUEST._serialized_end=1694
  _PROCESSOBSERVATIONREPLY._serialized_start=1697
  _PROCESSOBSERVATIONREPLY._serialized_end=1839
  _GETOBJECTBELIEFSREQUEST._serialized_start=1841
  _GETOBJECTBELIEFSREQUEST._serialized_end=1954
  _GETOBJECTBELIEFSREPLY._serialized_start=1957
  _GETOBJECTBELIEFSREPLY._serialized_end=2161
  _GETROBOTBELIEFREQUEST._serialized_start=2163
  _GETROBOTBELIEFREQUEST._serialized_end=2254
  _GETROBOTBELIEFREPLY._serialized_start=2257
  _GETROBOTBELIEFREPLY._serialized_end=2456
  _CREATEPLANNERREQUEST._serialized_start=2458
  _CREATEPLANNERREQUEST._serialized_end=2583
  _CREATEPLANNERREPLY._serialized_start=2586
  _CREATEPLANNERREPLY._serialized_end=2723
  _PLANACTIONREQUEST._serialized_start=2725
  _PLANACTIONREQUEST._serialized_end=2812
  _PLANACTIONREPLY._serialized_start=2815
  _PLANACTIONREPLY._serialized_end=3160
  _ACTIONFINISHEDREQUEST._serialized_start=3162
  _ACTIONFINISHEDREQUEST._serialized_end=3272
  _ACTIONFINISHEDREPLY._serialized_start=3275
  _ACTIONFINISHEDREPLY._serialized_end=3413
  _SEARCHREGIONPARAMS2D._serialized_start=3416
  _SEARCHREGIONPARAMS2D._serialized_end=3666
  _SEARCHREGIONPARAMS3D._serialized_start=3669
  _SEARCHREGIONPARAMS3D._serialized_end=3967
  _OBJECTBELIEF._serialized_start=3969
  _OBJECTBELIEF._serialized_end=4053
  _ROBOTBELIEF._serialized_start=4056
  _ROBOTBELIEF._serialized_end=4197
  _SLOOPOBJECTSEARCH._serialized_start=4200
  _SLOOPOBJECTSEARCH._serialized_end=5294
# @@protoc_insertion_point(module_scope)
