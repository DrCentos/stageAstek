# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from ...pipeline.v1.beam_runner_api_pb2 import (
    TestStreamPayload as org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___TestStreamPayload,
)

from typing import (
    Optional as typing___Optional,
    Text as typing___Text,
    Union as typing___Union,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int
if sys.version_info < (3,):
    builtin___buffer = buffer
    builtin___unicode = unicode


class TestStreamFileHeader(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    tag = ... # type: typing___Text

    def __init__(self,
        *,
        tag : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> TestStreamFileHeader: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> TestStreamFileHeader: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"tag"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"tag",b"tag"]) -> None: ...

class TestStreamFileRecord(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def recorded_event(self) -> org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___TestStreamPayload.Event: ...

    def __init__(self,
        *,
        recorded_event : typing___Optional[org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___TestStreamPayload.Event] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> TestStreamFileRecord: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> TestStreamFileRecord: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"recorded_event"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"recorded_event"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"recorded_event",b"recorded_event"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"recorded_event",b"recorded_event"]) -> None: ...