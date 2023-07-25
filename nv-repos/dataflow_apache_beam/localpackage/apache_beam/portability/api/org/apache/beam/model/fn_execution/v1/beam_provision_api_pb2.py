# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: org/apache/beam/model/fn_execution/v1/beam_provision_api.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ...pipeline.v1 import beam_runner_api_pb2 as org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_beam__runner__api__pb2
from ...pipeline.v1 import endpoints_pb2 as org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_endpoints__pb2
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='org/apache/beam/model/fn_execution/v1/beam_provision_api.proto',
  package='org.apache.beam.model.fn_execution.v1',
  syntax='proto3',
  serialized_options=b'\n$org.apache.beam.model.fnexecution.v1B\014ProvisionApiZNgithub.com/apache/beam/sdks/v2/go/pkg/beam/model/fnexecution_v1;fnexecution_v1',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n>org/apache/beam/model/fn_execution/v1/beam_provision_api.proto\x12%org.apache.beam.model.fn_execution.v1\x1a\x37org/apache/beam/model/pipeline/v1/beam_runner_api.proto\x1a\x31org/apache/beam/model/pipeline/v1/endpoints.proto\x1a\x1cgoogle/protobuf/struct.proto\"\x19\n\x17GetProvisionInfoRequest\"^\n\x18GetProvisionInfoResponse\x12\x42\n\x04info\x18\x01 \x01(\x0b\x32\x34.org.apache.beam.model.fn_execution.v1.ProvisionInfo\"\xb5\x05\n\rProvisionInfo\x12\x31\n\x10pipeline_options\x18\x03 \x01(\x0b\x32\x17.google.protobuf.Struct\x12\x17\n\x0fretrieval_token\x18\x06 \x01(\t\x12P\n\x0fstatus_endpoint\x18\x07 \x01(\x0b\x32\x37.org.apache.beam.model.pipeline.v1.ApiServiceDescriptor\x12Q\n\x10logging_endpoint\x18\x08 \x01(\x0b\x32\x37.org.apache.beam.model.pipeline.v1.ApiServiceDescriptor\x12R\n\x11\x61rtifact_endpoint\x18\t \x01(\x0b\x32\x37.org.apache.beam.model.pipeline.v1.ApiServiceDescriptor\x12Q\n\x10\x63ontrol_endpoint\x18\n \x01(\x0b\x32\x37.org.apache.beam.model.pipeline.v1.ApiServiceDescriptor\x12L\n\x0c\x64\x65pendencies\x18\x0b \x03(\x0b\x32\x36.org.apache.beam.model.pipeline.v1.ArtifactInformation\x12\x1b\n\x13runner_capabilities\x18\x0c \x03(\t\x12T\n\x08metadata\x18\r \x03(\x0b\x32\x42.org.apache.beam.model.fn_execution.v1.ProvisionInfo.MetadataEntry\x12\x1a\n\x12sibling_worker_ids\x18\x0e \x03(\t\x1a/\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x32\xa8\x01\n\x10ProvisionService\x12\x93\x01\n\x10GetProvisionInfo\x12>.org.apache.beam.model.fn_execution.v1.GetProvisionInfoRequest\x1a?.org.apache.beam.model.fn_execution.v1.GetProvisionInfoResponseB\x84\x01\n$org.apache.beam.model.fnexecution.v1B\x0cProvisionApiZNgithub.com/apache/beam/sdks/v2/go/pkg/beam/model/fnexecution_v1;fnexecution_v1b\x06proto3'
  ,
  dependencies=[org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_beam__runner__api__pb2.DESCRIPTOR,org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_endpoints__pb2.DESCRIPTOR,google_dot_protobuf_dot_struct__pb2.DESCRIPTOR,])




_GETPROVISIONINFOREQUEST = _descriptor.Descriptor(
  name='GetProvisionInfoRequest',
  full_name='org.apache.beam.model.fn_execution.v1.GetProvisionInfoRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=243,
  serialized_end=268,
)


_GETPROVISIONINFORESPONSE = _descriptor.Descriptor(
  name='GetProvisionInfoResponse',
  full_name='org.apache.beam.model.fn_execution.v1.GetProvisionInfoResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='info', full_name='org.apache.beam.model.fn_execution.v1.GetProvisionInfoResponse.info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=270,
  serialized_end=364,
)


_PROVISIONINFO_METADATAENTRY = _descriptor.Descriptor(
  name='MetadataEntry',
  full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.MetadataEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.MetadataEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.MetadataEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1013,
  serialized_end=1060,
)

_PROVISIONINFO = _descriptor.Descriptor(
  name='ProvisionInfo',
  full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='pipeline_options', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.pipeline_options', index=0,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='retrieval_token', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.retrieval_token', index=1,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status_endpoint', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.status_endpoint', index=2,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='logging_endpoint', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.logging_endpoint', index=3,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='artifact_endpoint', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.artifact_endpoint', index=4,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='control_endpoint', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.control_endpoint', index=5,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dependencies', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.dependencies', index=6,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='runner_capabilities', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.runner_capabilities', index=7,
      number=12, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.metadata', index=8,
      number=13, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sibling_worker_ids', full_name='org.apache.beam.model.fn_execution.v1.ProvisionInfo.sibling_worker_ids', index=9,
      number=14, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_PROVISIONINFO_METADATAENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=367,
  serialized_end=1060,
)

_GETPROVISIONINFORESPONSE.fields_by_name['info'].message_type = _PROVISIONINFO
_PROVISIONINFO_METADATAENTRY.containing_type = _PROVISIONINFO
_PROVISIONINFO.fields_by_name['pipeline_options'].message_type = google_dot_protobuf_dot_struct__pb2._STRUCT
_PROVISIONINFO.fields_by_name['status_endpoint'].message_type = org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_endpoints__pb2._APISERVICEDESCRIPTOR
_PROVISIONINFO.fields_by_name['logging_endpoint'].message_type = org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_endpoints__pb2._APISERVICEDESCRIPTOR
_PROVISIONINFO.fields_by_name['artifact_endpoint'].message_type = org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_endpoints__pb2._APISERVICEDESCRIPTOR
_PROVISIONINFO.fields_by_name['control_endpoint'].message_type = org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_endpoints__pb2._APISERVICEDESCRIPTOR
_PROVISIONINFO.fields_by_name['dependencies'].message_type = org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_beam__runner__api__pb2._ARTIFACTINFORMATION
_PROVISIONINFO.fields_by_name['metadata'].message_type = _PROVISIONINFO_METADATAENTRY
DESCRIPTOR.message_types_by_name['GetProvisionInfoRequest'] = _GETPROVISIONINFOREQUEST
DESCRIPTOR.message_types_by_name['GetProvisionInfoResponse'] = _GETPROVISIONINFORESPONSE
DESCRIPTOR.message_types_by_name['ProvisionInfo'] = _PROVISIONINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GetProvisionInfoRequest = _reflection.GeneratedProtocolMessageType('GetProvisionInfoRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETPROVISIONINFOREQUEST,
  '__module__' : 'org.apache.beam.model.fn_execution.v1.beam_provision_api_pb2'
  # @@protoc_insertion_point(class_scope:org.apache.beam.model.fn_execution.v1.GetProvisionInfoRequest)
  })
_sym_db.RegisterMessage(GetProvisionInfoRequest)

GetProvisionInfoResponse = _reflection.GeneratedProtocolMessageType('GetProvisionInfoResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETPROVISIONINFORESPONSE,
  '__module__' : 'org.apache.beam.model.fn_execution.v1.beam_provision_api_pb2'
  # @@protoc_insertion_point(class_scope:org.apache.beam.model.fn_execution.v1.GetProvisionInfoResponse)
  })
_sym_db.RegisterMessage(GetProvisionInfoResponse)

ProvisionInfo = _reflection.GeneratedProtocolMessageType('ProvisionInfo', (_message.Message,), {

  'MetadataEntry' : _reflection.GeneratedProtocolMessageType('MetadataEntry', (_message.Message,), {
    'DESCRIPTOR' : _PROVISIONINFO_METADATAENTRY,
    '__module__' : 'org.apache.beam.model.fn_execution.v1.beam_provision_api_pb2'
    # @@protoc_insertion_point(class_scope:org.apache.beam.model.fn_execution.v1.ProvisionInfo.MetadataEntry)
    })
  ,
  'DESCRIPTOR' : _PROVISIONINFO,
  '__module__' : 'org.apache.beam.model.fn_execution.v1.beam_provision_api_pb2'
  # @@protoc_insertion_point(class_scope:org.apache.beam.model.fn_execution.v1.ProvisionInfo)
  })
_sym_db.RegisterMessage(ProvisionInfo)
_sym_db.RegisterMessage(ProvisionInfo.MetadataEntry)


DESCRIPTOR._options = None
_PROVISIONINFO_METADATAENTRY._options = None

_PROVISIONSERVICE = _descriptor.ServiceDescriptor(
  name='ProvisionService',
  full_name='org.apache.beam.model.fn_execution.v1.ProvisionService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1063,
  serialized_end=1231,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetProvisionInfo',
    full_name='org.apache.beam.model.fn_execution.v1.ProvisionService.GetProvisionInfo',
    index=0,
    containing_service=None,
    input_type=_GETPROVISIONINFOREQUEST,
    output_type=_GETPROVISIONINFORESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_PROVISIONSERVICE)

DESCRIPTOR.services_by_name['ProvisionService'] = _PROVISIONSERVICE

# @@protoc_insertion_point(module_scope)