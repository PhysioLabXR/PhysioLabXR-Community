import RPCTest_pb2_grpc, RPCTest_pb2

class RPCExampleServer(RPCTest_pb2_grpc.RPCTestServicer):

    script_instance = None

    def TestRPCOneArgOneReturn(self, request, context):
        result = self.script_instance.PredictFromInput(*request)
        return RPCTest_pb2.TestRPCOneArgOneReturnResponse(*result)