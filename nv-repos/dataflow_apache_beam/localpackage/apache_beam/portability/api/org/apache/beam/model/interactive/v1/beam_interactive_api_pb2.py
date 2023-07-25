# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: org/apache/beam/model/interactive/v1/beam_interactive_api.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from ...pipeline.v1 import beam_runner_api_pb2 as org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_beam__runner__api__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='org/apache/beam/model/interactive/v1/beam_interactive_api.proto',
  package='org.apache.beam.model.interactive.v1',
  syntax='proto3',
  serialized_options=b'\n$org.apache.beam.model.interactive.v1B\016InteractiveApiZ\016interactive_v1',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n?org/apache/beam/model/interactive/v1/beam_interactive_api.proto\x12$org.apache.beam.model.interactive.v1\x1a\x37org/apache/beam/model/pipeline/v1/beam_runner_api.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"#\n\x14TestStreamFileHeader\x12\x0b\n\x03tag\x18\x01 \x01(\t\"j\n\x14TestStreamFileRecord\x12R\n\x0erecorded_event\x18\x01 \x01(\x0b\x32:.org.apache.beam.model.pipeline.v1.TestStreamPayload.EventBF\n$org.apache.beam.model.interactive.v1B\x0eInteractiveApiZ\x0einteractive_v1b\x06proto3'
  ,
  dependencies=[org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_beam__runner__api__pb2.DESCRIPTOR,google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])




_TESTSTREAMFILEHEADER = _descriptor.Descriptor(
  name='TestStreamFileHeader',
  full_name='org.apache.beam.model.interactive.v1.TestStreamFileHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='tag', full_name='org.apache.beam.model.interactive.v1.TestStreamFileHeader.tag', index=0,
      number=1, type=9, cpp_type=9, label=1,
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
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=195,
  serialized_end=230,
)


_TESTSTREAMFILERECORD = _descriptor.Descriptor(
  name='TestStreamFileRecord',
  full_name='org.apache.beam.model.interactive.v1.TestStreamFileRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='recorded_event', full_name='org.apache.beam.model.interactive.v1.TestStreamFileRecord.recorded_event', index=0,
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
  serialized_start=232,
  serialized_end=338,
)

_TESTSTREAMFILERECORD.fields_by_name['recorded_event'].message_type = org_dot_apache_dot_beam_dot_model_dot_pipeline_dot_v1_dot_beam__runner__api__pb2._TESTSTREAMPAYLOAD_EVENT
DESCRIPTOR.message_types_by_name['TestStreamFileHeader'] = _TESTSTREAMFILEHEADER
DESCRIPTOR.message_types_by_name['TestStreamFileRecord'] = _TESTSTREAMFILERECORD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TestStreamFileHeader = _reflection.GeneratedProtocolMessageType('TestStreamFileHeader', (_message.Message,), {
  'DESCRIPTOR' : _TESTSTREAMFILEHEADER,
  '__module__' : 'org.apache.beam.model.interactive.v1.beam_interactive_api_pb2'
  # @@protoc_insertion_point(class_scope:org.apache.beam.model.interactive.v1.TestStreamFileHeader)
  })
_sym_db.RegisterMessage(TestStreamFileHeader)

TestStreamFileRecord = _reflection.GeneratedProtocolMessageType('TestStreamFileRecord', (_message.Message,), {
  'DESCRIPTOR' : _TESTSTREAMFILERECORD,
  '__module__' : 'org.apache.beam.model.interactive.v1.beam_interactive_api_pb2'
  # @@protoc_insertion_point(class_scope:org.apache.beam.model.interactive.v1.TestStreamFileRecord)
  })
_sym_db.RegisterMessage(TestStreamFileRecord)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
