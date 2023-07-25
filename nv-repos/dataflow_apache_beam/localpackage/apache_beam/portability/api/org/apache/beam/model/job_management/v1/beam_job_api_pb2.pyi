# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from google.protobuf.struct_pb2 import (
    Struct as google___protobuf___struct_pb2___Struct,
)

from google.protobuf.timestamp_pb2 import (
    Timestamp as google___protobuf___timestamp_pb2___Timestamp,
)

from ...pipeline.v1.beam_runner_api_pb2 import (
    Pipeline as org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___Pipeline,
)

from ...pipeline.v1.endpoints_pb2 import (
    ApiServiceDescriptor as org___apache___beam___model___pipeline___v1___endpoints_pb2___ApiServiceDescriptor,
)

from ...pipeline.v1.metrics_pb2 import (
    MonitoringInfo as org___apache___beam___model___pipeline___v1___metrics_pb2___MonitoringInfo,
)

from typing import (
    Iterable as typing___Iterable,
    List as typing___List,
    Optional as typing___Optional,
    Text as typing___Text,
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


class PrepareJobRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_name = ... # type: typing___Text

    @property
    def pipeline(self) -> org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___Pipeline: ...

    @property
    def pipeline_options(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        pipeline : typing___Optional[org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___Pipeline] = None,
        pipeline_options : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        job_name : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> PrepareJobRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> PrepareJobRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"pipeline",u"pipeline_options"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"job_name",u"pipeline",u"pipeline_options"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"pipeline",b"pipeline",u"pipeline_options",b"pipeline_options"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"job_name",b"job_name",u"pipeline",b"pipeline",u"pipeline_options",b"pipeline_options"]) -> None: ...

class PrepareJobResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    preparation_id = ... # type: typing___Text
    staging_session_token = ... # type: typing___Text

    @property
    def artifact_staging_endpoint(self) -> org___apache___beam___model___pipeline___v1___endpoints_pb2___ApiServiceDescriptor: ...

    def __init__(self,
        *,
        preparation_id : typing___Optional[typing___Text] = None,
        artifact_staging_endpoint : typing___Optional[org___apache___beam___model___pipeline___v1___endpoints_pb2___ApiServiceDescriptor] = None,
        staging_session_token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> PrepareJobResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> PrepareJobResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"artifact_staging_endpoint"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"artifact_staging_endpoint",u"preparation_id",u"staging_session_token"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"artifact_staging_endpoint",b"artifact_staging_endpoint"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"artifact_staging_endpoint",b"artifact_staging_endpoint",u"preparation_id",b"preparation_id",u"staging_session_token",b"staging_session_token"]) -> None: ...

class RunJobRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    preparation_id = ... # type: typing___Text
    retrieval_token = ... # type: typing___Text

    def __init__(self,
        *,
        preparation_id : typing___Optional[typing___Text] = None,
        retrieval_token : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> RunJobRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> RunJobRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"preparation_id",u"retrieval_token"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"preparation_id",b"preparation_id",u"retrieval_token",b"retrieval_token"]) -> None: ...

class RunJobResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> RunJobResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> RunJobResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id"]) -> None: ...

class CancelJobRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> CancelJobRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> CancelJobRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id"]) -> None: ...

class CancelJobResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    state = ... # type: JobState.Enum

    def __init__(self,
        *,
        state : typing___Optional[JobState.Enum] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> CancelJobResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> CancelJobResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"state"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"state",b"state"]) -> None: ...

class JobInfo(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text
    job_name = ... # type: typing___Text
    state = ... # type: JobState.Enum

    @property
    def pipeline_options(self) -> google___protobuf___struct_pb2___Struct: ...

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        job_name : typing___Optional[typing___Text] = None,
        pipeline_options : typing___Optional[google___protobuf___struct_pb2___Struct] = None,
        state : typing___Optional[JobState.Enum] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> JobInfo: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> JobInfo: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"pipeline_options"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",u"job_name",u"pipeline_options",u"state"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"pipeline_options",b"pipeline_options"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id",u"job_name",b"job_name",u"pipeline_options",b"pipeline_options",u"state",b"state"]) -> None: ...

class GetJobsRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobsRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobsRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class GetJobsResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def job_info(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[JobInfo]: ...

    def __init__(self,
        *,
        job_info : typing___Optional[typing___Iterable[JobInfo]] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobsResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobsResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_info"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_info",b"job_info"]) -> None: ...

class GetJobStateRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobStateRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobStateRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id"]) -> None: ...

class JobStateEvent(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    state = ... # type: JobState.Enum

    @property
    def timestamp(self) -> google___protobuf___timestamp_pb2___Timestamp: ...

    def __init__(self,
        *,
        state : typing___Optional[JobState.Enum] = None,
        timestamp : typing___Optional[google___protobuf___timestamp_pb2___Timestamp] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> JobStateEvent: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> JobStateEvent: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"state",u"timestamp"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"timestamp",b"timestamp"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"state",b"state",u"timestamp",b"timestamp"]) -> None: ...

class GetJobPipelineRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobPipelineRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobPipelineRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id"]) -> None: ...

class GetJobPipelineResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def pipeline(self) -> org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___Pipeline: ...

    def __init__(self,
        *,
        pipeline : typing___Optional[org___apache___beam___model___pipeline___v1___beam_runner_api_pb2___Pipeline] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobPipelineResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobPipelineResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"pipeline"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"pipeline"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"pipeline",b"pipeline"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"pipeline",b"pipeline"]) -> None: ...

class JobMessagesRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> JobMessagesRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> JobMessagesRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id"]) -> None: ...

class JobMessage(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class MessageImportance(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'JobMessage.MessageImportance': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['JobMessage.MessageImportance']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'JobMessage.MessageImportance']]: ...
        MESSAGE_IMPORTANCE_UNSPECIFIED = typing___cast('JobMessage.MessageImportance', 0)
        JOB_MESSAGE_DEBUG = typing___cast('JobMessage.MessageImportance', 1)
        JOB_MESSAGE_DETAILED = typing___cast('JobMessage.MessageImportance', 2)
        JOB_MESSAGE_BASIC = typing___cast('JobMessage.MessageImportance', 3)
        JOB_MESSAGE_WARNING = typing___cast('JobMessage.MessageImportance', 4)
        JOB_MESSAGE_ERROR = typing___cast('JobMessage.MessageImportance', 5)
    MESSAGE_IMPORTANCE_UNSPECIFIED = typing___cast('JobMessage.MessageImportance', 0)
    JOB_MESSAGE_DEBUG = typing___cast('JobMessage.MessageImportance', 1)
    JOB_MESSAGE_DETAILED = typing___cast('JobMessage.MessageImportance', 2)
    JOB_MESSAGE_BASIC = typing___cast('JobMessage.MessageImportance', 3)
    JOB_MESSAGE_WARNING = typing___cast('JobMessage.MessageImportance', 4)
    JOB_MESSAGE_ERROR = typing___cast('JobMessage.MessageImportance', 5)

    message_id = ... # type: typing___Text
    time = ... # type: typing___Text
    importance = ... # type: JobMessage.MessageImportance
    message_text = ... # type: typing___Text

    def __init__(self,
        *,
        message_id : typing___Optional[typing___Text] = None,
        time : typing___Optional[typing___Text] = None,
        importance : typing___Optional[JobMessage.MessageImportance] = None,
        message_text : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> JobMessage: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> JobMessage: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"importance",u"message_id",u"message_text",u"time"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"importance",b"importance",u"message_id",b"message_id",u"message_text",b"message_text",u"time",b"time"]) -> None: ...

class JobMessagesResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def message_response(self) -> JobMessage: ...

    @property
    def state_response(self) -> JobStateEvent: ...

    def __init__(self,
        *,
        message_response : typing___Optional[JobMessage] = None,
        state_response : typing___Optional[JobStateEvent] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> JobMessagesResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> JobMessagesResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"message_response",u"response",u"state_response"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"message_response",u"response",u"state_response"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"message_response",b"message_response",u"response",b"response",u"state_response",b"state_response"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"message_response",b"message_response",u"response",b"response",u"state_response",b"state_response"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions___Literal[u"response",b"response"]) -> typing_extensions___Literal["message_response","state_response"]: ...

class JobState(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class Enum(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'JobState.Enum': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['JobState.Enum']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'JobState.Enum']]: ...
        UNSPECIFIED = typing___cast('JobState.Enum', 0)
        STOPPED = typing___cast('JobState.Enum', 1)
        RUNNING = typing___cast('JobState.Enum', 2)
        DONE = typing___cast('JobState.Enum', 3)
        FAILED = typing___cast('JobState.Enum', 4)
        CANCELLED = typing___cast('JobState.Enum', 5)
        UPDATED = typing___cast('JobState.Enum', 6)
        DRAINING = typing___cast('JobState.Enum', 7)
        DRAINED = typing___cast('JobState.Enum', 8)
        STARTING = typing___cast('JobState.Enum', 9)
        CANCELLING = typing___cast('JobState.Enum', 10)
        UPDATING = typing___cast('JobState.Enum', 11)
    UNSPECIFIED = typing___cast('JobState.Enum', 0)
    STOPPED = typing___cast('JobState.Enum', 1)
    RUNNING = typing___cast('JobState.Enum', 2)
    DONE = typing___cast('JobState.Enum', 3)
    FAILED = typing___cast('JobState.Enum', 4)
    CANCELLED = typing___cast('JobState.Enum', 5)
    UPDATED = typing___cast('JobState.Enum', 6)
    DRAINING = typing___cast('JobState.Enum', 7)
    DRAINED = typing___cast('JobState.Enum', 8)
    STARTING = typing___cast('JobState.Enum', 9)
    CANCELLING = typing___cast('JobState.Enum', 10)
    UPDATING = typing___cast('JobState.Enum', 11)


    def __init__(self,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> JobState: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> JobState: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class GetJobMetricsRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    job_id = ... # type: typing___Text

    def __init__(self,
        *,
        job_id : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobMetricsRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobMetricsRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"job_id",b"job_id"]) -> None: ...

class GetJobMetricsResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def metrics(self) -> MetricResults: ...

    def __init__(self,
        *,
        metrics : typing___Optional[MetricResults] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> GetJobMetricsResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> GetJobMetricsResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"metrics"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"metrics"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"metrics",b"metrics"]) -> builtin___bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"metrics",b"metrics"]) -> None: ...

class MetricResults(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def attempted(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[org___apache___beam___model___pipeline___v1___metrics_pb2___MonitoringInfo]: ...

    @property
    def committed(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[org___apache___beam___model___pipeline___v1___metrics_pb2___MonitoringInfo]: ...

    def __init__(self,
        *,
        attempted : typing___Optional[typing___Iterable[org___apache___beam___model___pipeline___v1___metrics_pb2___MonitoringInfo]] = None,
        committed : typing___Optional[typing___Iterable[org___apache___beam___model___pipeline___v1___metrics_pb2___MonitoringInfo]] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> MetricResults: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> MetricResults: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"attempted",u"committed"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"attempted",b"attempted",u"committed",b"committed"]) -> None: ...

class DescribePipelineOptionsRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> DescribePipelineOptionsRequest: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> DescribePipelineOptionsRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class PipelineOptionType(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class Enum(builtin___int):
        DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
        @classmethod
        def Name(cls, number: builtin___int) -> builtin___str: ...
        @classmethod
        def Value(cls, name: builtin___str) -> 'PipelineOptionType.Enum': ...
        @classmethod
        def keys(cls) -> typing___List[builtin___str]: ...
        @classmethod
        def values(cls) -> typing___List['PipelineOptionType.Enum']: ...
        @classmethod
        def items(cls) -> typing___List[typing___Tuple[builtin___str, 'PipelineOptionType.Enum']]: ...
        STRING = typing___cast('PipelineOptionType.Enum', 0)
        BOOLEAN = typing___cast('PipelineOptionType.Enum', 1)
        INTEGER = typing___cast('PipelineOptionType.Enum', 2)
        NUMBER = typing___cast('PipelineOptionType.Enum', 3)
        ARRAY = typing___cast('PipelineOptionType.Enum', 4)
        OBJECT = typing___cast('PipelineOptionType.Enum', 5)
    STRING = typing___cast('PipelineOptionType.Enum', 0)
    BOOLEAN = typing___cast('PipelineOptionType.Enum', 1)
    INTEGER = typing___cast('PipelineOptionType.Enum', 2)
    NUMBER = typing___cast('PipelineOptionType.Enum', 3)
    ARRAY = typing___cast('PipelineOptionType.Enum', 4)
    OBJECT = typing___cast('PipelineOptionType.Enum', 5)


    def __init__(self,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> PipelineOptionType: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> PipelineOptionType: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...

class PipelineOptionDescriptor(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    name = ... # type: typing___Text
    type = ... # type: PipelineOptionType.Enum
    description = ... # type: typing___Text
    default_value = ... # type: typing___Text
    group = ... # type: typing___Text

    def __init__(self,
        *,
        name : typing___Optional[typing___Text] = None,
        type : typing___Optional[PipelineOptionType.Enum] = None,
        description : typing___Optional[typing___Text] = None,
        default_value : typing___Optional[typing___Text] = None,
        group : typing___Optional[typing___Text] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> PipelineOptionDescriptor: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> PipelineOptionDescriptor: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"default_value",u"description",u"group",u"name",u"type"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"default_value",b"default_value",u"description",b"description",u"group",b"group",u"name",b"name",u"type",b"type"]) -> None: ...

class DescribePipelineOptionsResponse(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def options(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[PipelineOptionDescriptor]: ...

    def __init__(self,
        *,
        options : typing___Optional[typing___Iterable[PipelineOptionDescriptor]] = None,
        ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> DescribePipelineOptionsResponse: ...
    else:
        @classmethod
        def FromString(cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]) -> DescribePipelineOptionsResponse: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"options"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"options",b"options"]) -> None: ...