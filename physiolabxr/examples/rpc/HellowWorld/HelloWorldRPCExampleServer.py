from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import HelloWorldRPCExample_pb2_grpc, HelloWorldRPCExample_pb2

class HelloWorldRPCServer(HelloWorldRPCExample_pb2_grpc.HelloWorldRPCServicer):
    script_instance = None
    def SayHello(self, request, context):
        result = self.script_instance.SayHello(**MessageToDict(request))
        return HelloWorldRPCExample_pb2.SayHelloResponse(message=result)
