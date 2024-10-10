from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import AsyncRPCExample_pb2_grpc, AsyncRPCExample_pb2

class AsyncRPCExampleServer(AsyncRPCExample_pb2_grpc.AsyncRPCExampleServicer):
    script_instance = None
    async def AsyncOneArgOneReturn(self, request, context):
        result = self.script_instance.AsyncOneArgOneReturn(**MessageToDict(request))
        return AsyncRPCExample_pb2.AsyncOneArgOneReturnResponse(message=result)
