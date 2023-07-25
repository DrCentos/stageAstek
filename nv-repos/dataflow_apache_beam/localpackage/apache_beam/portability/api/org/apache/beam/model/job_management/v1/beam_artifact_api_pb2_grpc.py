# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import beam_artifact_api_pb2 as org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2


class ArtifactRetrievalServiceStub(object):
    """A service to retrieve artifacts for use in a Job.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ResolveArtifacts = channel.unary_unary(
                '/org.apache.beam.model.job_management.v1.ArtifactRetrievalService/ResolveArtifacts',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ResolveArtifactsRequest.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ResolveArtifactsResponse.FromString,
                )
        self.GetArtifact = channel.unary_stream(
                '/org.apache.beam.model.job_management.v1.ArtifactRetrievalService/GetArtifact',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetArtifactRequest.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetArtifactResponse.FromString,
                )


class ArtifactRetrievalServiceServicer(object):
    """A service to retrieve artifacts for use in a Job.
    """

    def ResolveArtifacts(self, request, context):
        """Resolves the given artifact references into one or more replacement
        artifact references (e.g. a Maven dependency into a (transitive) set
        of jars.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetArtifact(self, request, context):
        """Retrieves the given artifact as a stream of bytes.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ArtifactRetrievalServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ResolveArtifacts': grpc.unary_unary_rpc_method_handler(
                    servicer.ResolveArtifacts,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ResolveArtifactsRequest.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ResolveArtifactsResponse.SerializeToString,
            ),
            'GetArtifact': grpc.unary_stream_rpc_method_handler(
                    servicer.GetArtifact,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetArtifactRequest.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetArtifactResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.apache.beam.model.job_management.v1.ArtifactRetrievalService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ArtifactRetrievalService(object):
    """A service to retrieve artifacts for use in a Job.
    """

    @staticmethod
    def ResolveArtifacts(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.apache.beam.model.job_management.v1.ArtifactRetrievalService/ResolveArtifacts',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ResolveArtifactsRequest.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ResolveArtifactsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetArtifact(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/org.apache.beam.model.job_management.v1.ArtifactRetrievalService/GetArtifact',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetArtifactRequest.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetArtifactResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ArtifactStagingServiceStub(object):
    """A service that allows the client to act as an ArtifactRetrievalService,
    for a particular job with the server initiating requests and receiving
    responses.

    A client calls the service with an ArtifactResponseWrapper that has the
    staging token set, and thereafter responds to the server's requests.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ReverseArtifactRetrievalService = channel.stream_stream(
                '/org.apache.beam.model.job_management.v1.ArtifactStagingService/ReverseArtifactRetrievalService',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactResponseWrapper.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactRequestWrapper.FromString,
                )


class ArtifactStagingServiceServicer(object):
    """A service that allows the client to act as an ArtifactRetrievalService,
    for a particular job with the server initiating requests and receiving
    responses.

    A client calls the service with an ArtifactResponseWrapper that has the
    staging token set, and thereafter responds to the server's requests.
    """

    def ReverseArtifactRetrievalService(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ArtifactStagingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ReverseArtifactRetrievalService': grpc.stream_stream_rpc_method_handler(
                    servicer.ReverseArtifactRetrievalService,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactResponseWrapper.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactRequestWrapper.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.apache.beam.model.job_management.v1.ArtifactStagingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ArtifactStagingService(object):
    """A service that allows the client to act as an ArtifactRetrievalService,
    for a particular job with the server initiating requests and receiving
    responses.

    A client calls the service with an ArtifactResponseWrapper that has the
    staging token set, and thereafter responds to the server's requests.
    """

    @staticmethod
    def ReverseArtifactRetrievalService(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/org.apache.beam.model.job_management.v1.ArtifactStagingService/ReverseArtifactRetrievalService',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactResponseWrapper.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactRequestWrapper.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class LegacyArtifactStagingServiceStub(object):
    """Legacy artifact staging service for pipeline-level artifacts.

    A service to stage artifacts for use in a Job.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PutArtifact = channel.stream_unary(
                '/org.apache.beam.model.job_management.v1.LegacyArtifactStagingService/PutArtifact',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.PutArtifactRequest.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.PutArtifactResponse.FromString,
                )
        self.CommitManifest = channel.unary_unary(
                '/org.apache.beam.model.job_management.v1.LegacyArtifactStagingService/CommitManifest',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.CommitManifestRequest.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.CommitManifestResponse.FromString,
                )


class LegacyArtifactStagingServiceServicer(object):
    """Legacy artifact staging service for pipeline-level artifacts.

    A service to stage artifacts for use in a Job.
    """

    def PutArtifact(self, request_iterator, context):
        """Stage an artifact to be available during job execution. The first request must contain the
        name of the artifact. All future requests must contain sequential chunks of the content of
        the artifact.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CommitManifest(self, request, context):
        """Commit the manifest for a Job. All artifacts must have been successfully uploaded
        before this call is made.

        Throws error INVALID_ARGUMENT if not all of the members of the manifest are present
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LegacyArtifactStagingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'PutArtifact': grpc.stream_unary_rpc_method_handler(
                    servicer.PutArtifact,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.PutArtifactRequest.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.PutArtifactResponse.SerializeToString,
            ),
            'CommitManifest': grpc.unary_unary_rpc_method_handler(
                    servicer.CommitManifest,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.CommitManifestRequest.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.CommitManifestResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.apache.beam.model.job_management.v1.LegacyArtifactStagingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LegacyArtifactStagingService(object):
    """Legacy artifact staging service for pipeline-level artifacts.

    A service to stage artifacts for use in a Job.
    """

    @staticmethod
    def PutArtifact(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/org.apache.beam.model.job_management.v1.LegacyArtifactStagingService/PutArtifact',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.PutArtifactRequest.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.PutArtifactResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CommitManifest(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.apache.beam.model.job_management.v1.LegacyArtifactStagingService/CommitManifest',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.CommitManifestRequest.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.CommitManifestResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class LegacyArtifactRetrievalServiceStub(object):
    """A service to retrieve artifacts for use in a Job.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetManifest = channel.unary_unary(
                '/org.apache.beam.model.job_management.v1.LegacyArtifactRetrievalService/GetManifest',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetManifestRequest.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetManifestResponse.FromString,
                )
        self.GetArtifact = channel.unary_stream(
                '/org.apache.beam.model.job_management.v1.LegacyArtifactRetrievalService/GetArtifact',
                request_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.LegacyGetArtifactRequest.SerializeToString,
                response_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactChunk.FromString,
                )


class LegacyArtifactRetrievalServiceServicer(object):
    """A service to retrieve artifacts for use in a Job.
    """

    def GetManifest(self, request, context):
        """Get the manifest for the job
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetArtifact(self, request, context):
        """Get an artifact staged for the job. The requested artifact must be within the manifest
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LegacyArtifactRetrievalServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetManifest': grpc.unary_unary_rpc_method_handler(
                    servicer.GetManifest,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetManifestRequest.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetManifestResponse.SerializeToString,
            ),
            'GetArtifact': grpc.unary_stream_rpc_method_handler(
                    servicer.GetArtifact,
                    request_deserializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.LegacyGetArtifactRequest.FromString,
                    response_serializer=org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactChunk.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.apache.beam.model.job_management.v1.LegacyArtifactRetrievalService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LegacyArtifactRetrievalService(object):
    """A service to retrieve artifacts for use in a Job.
    """

    @staticmethod
    def GetManifest(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.apache.beam.model.job_management.v1.LegacyArtifactRetrievalService/GetManifest',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetManifestRequest.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.GetManifestResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetArtifact(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/org.apache.beam.model.job_management.v1.LegacyArtifactRetrievalService/GetArtifact',
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.LegacyGetArtifactRequest.SerializeToString,
            org_dot_apache_dot_beam_dot_model_dot_job__management_dot_v1_dot_beam__artifact__api__pb2.ArtifactChunk.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
