from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import NeuroCooked_pb2_grpc, NeuroCooked_pb2

class NeuroCookedServer(NeuroCooked_pb2_grpc.NeuroCookedServicer):
    script_instance = None
    async def add_seq_data(self, request, context):
        result = self.script_instance.add_seq_data(**MessageToDict(request))
        return empty_pb2.Empty()

    async def decode(self, request, context):
        result = self.script_instance.decode()
        return NeuroCooked_pb2.decodeResponse(message=result)
