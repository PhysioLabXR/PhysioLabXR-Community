from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict

import RPCTest_pb2_grpc, RPCTest_pb2

class RPCTestServer(RPCTest_pb2_grpc.RPCTestServicer):

    script_instance = None

    def TestRPCOneArgOneReturn(self, request, context):
        result = self.script_instance.TestRPCOneArgOneReturn(**MessageToDict(request))
        return RPCTest_pb2.TestRPCOneArgOneReturnResponse(message=result)

    def TestRPCTwoArgTwoReturn(self, request, context):
        result = self.script_instance.TestRPCTwoArgTwoReturn(**MessageToDict(request))
        return RPCTest_pb2.TestRPCTwoArgTwoReturnResponse(message0=result[0], message1=result[1])

    def TestRPCNoInputNoReturn(self, request, context):
        result = self.script_instance.TestRPCNoInputNoReturn()
        return empty_pb2.Empty()

    def TestRPCNoReturn(self, request, context):
        result = self.script_instance.TestRPCNoReturn(**MessageToDict(request))
        return empty_pb2.Empty()

    def TestRPCNoArgs(self, request, context):
        result = self.script_instance.TestRPCNoArgs()
        return RPCTest_pb2.TestRPCNoArgsResponse(message=result)
