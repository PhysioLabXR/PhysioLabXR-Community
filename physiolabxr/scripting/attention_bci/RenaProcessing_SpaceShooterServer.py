from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import RenaProcessing_SpaceShooter_pb2_grpc, RenaProcessing_SpaceShooter_pb2

class RenaProcessingServer(RenaProcessing_SpaceShooter_pb2_grpc.RenaProcessingServicer):
    script_instance = None
    async def add_block_data(self, request, context):
        result = self.script_instance.add_block_data()
        return empty_pb2.Empty()
