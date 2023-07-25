# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
)

from google.protobuf.duration_pb2 import (
    Duration as google___protobuf___duration_pb2___Duration,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from google.protobuf.timestamp_pb2 import (
    Timestamp as google___protobuf___timestamp_pb2___Timestamp,
)

from typing import (
    List as typing___List,
    Optional as typing___Optional,
    Tuple as typing___Tuple,
    Union as typing___Union,
    cast as typing___cast,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int
builtin___str = str
if sys.version_info < (3,):
    builtin___buffer = buffer
    builtin___unicode = unicode


class GlobalWindowsPayload(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class Enum(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'GlobalWindowsPayload.Enum': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['GlobalWindowsPayload.Enum']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'GlobalWindowsPayload.Enum']]: ...
        PROPERTIES = typing___cast('GlobalWindowsPayload.Enum', 0)
    PROPERTIES = typing___cast('GlobalWindowsPayload.Enum', 0)


    def __init__(self,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GlobalWindowsPayload: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GlobalWindowsPayload: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class FixedWindowsPayload(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class Enum(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'FixedWindowsPayload.Enum': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['FixedWindowsPayload.Enum']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'FixedWindowsPayload.Enum']]: ...
        PROPERTIES = typing___cast('FixedWindowsPayload.Enum', 0)
    PROPERTIES = typing___cast('FixedWindowsPayload.Enum', 0)


    @property
    def size(self) -> google___protobuf___duration_pb2___Duration: ...

    @property
    def offset(self) -> google___protobuf___timestamp_pb2___Timestamp: ...

    def __init__(self,
        *,
        size : typing___Optional[google___protobuf___duration_pb2___Duration] = None,
        offset : typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> FixedWindowsPayload: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> FixedWindowsPayload: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"offset",u"size"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"offset",u"size"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"offset",b"offset",u"size",b"size"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"offset",b"offset",u"size",b"size"]) -> None: ...

class SlidingWindowsPayload(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class Enum(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'SlidingWindowsPayload.Enum': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['SlidingWindowsPayload.Enum']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'SlidingWindowsPayload.Enum']]: ...
        PROPERTIES = typing___cast('SlidingWindowsPayload.Enum', 0)
    PROPERTIES = typing___cast('SlidingWindowsPayload.Enum', 0)


    @property
    def size(self) -> google___protobuf___duration_pb2___Duration: ...

    @property
    def offset(self) -> google___protobuf___timestamp_pb2___Timestamp: ...

    @property
    def period(self) -> google___protobuf___duration_pb2___Duration: ...

    def __init__(self,
        *,
        size : typing___Optional[google___protobuf___duration_pb2___Duration] = None,
        offset : typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        period : typing___Optional[google___protobuf___duration_pb2___Duration] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> SlidingWindowsPayload: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> SlidingWindowsPayload: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"offset",u"period",u"size"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"offset",u"period",u"size"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"offset",b"offset",u"period",b"period",u"size",b"size"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"offset",b"offset",u"period",b"period",u"size",b"size"]) -> None: ...

class SessionWindowsPayload(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class Enum(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'SessionWindowsPayload.Enum': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['SessionWindowsPayload.Enum']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'SessionWindowsPayload.Enum']]: ...
        PROPERTIES = typing___cast('SessionWindowsPayload.Enum', 0)
    PROPERTIES = typing___cast('SessionWindowsPayload.Enum', 0)


    @property
    def gap_size(self) -> google___protobuf___duration_pb2___Duration: ...

    def __init__(self,
        *,
        gap_size : typing___Optional[google___protobuf___duration_pb2___Duration] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> SessionWindowsPayload: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> SessionWindowsPayload: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"gap_size"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"gap_size"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"gap_size",b"gap_size"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"gap_size",b"gap_size"]) -> None: ...