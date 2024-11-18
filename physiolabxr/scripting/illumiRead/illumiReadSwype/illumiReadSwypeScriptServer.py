from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import illumiReadSwypeScript_pb2_grpc, illumiReadSwypeScript_pb2

class IllumiReadSwypeScriptServer(illumiReadSwypeScript_pb2_grpc.IllumiReadSwypeScriptServicer):
    script_instance = None
    async def ContextRPC(self, request, context):
        result = self.script_instance.ContextRPC(**MessageToDict(request))
        return illumiReadSwypeScript_pb2.ContextRPCResponse(message=result)

    async def Tap2CharRPC(self, request, context):
        result = self.script_instance.Tap2CharRPC(**MessageToDict(request))
        return illumiReadSwypeScript_pb2.Tap2CharRPCResponse(message=result)
