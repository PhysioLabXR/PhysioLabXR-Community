import grpc

from physiolabxr.examples.rpc import RPCExample_pb2_grpc, RPCExample_pb2

channel = grpc.insecure_channel('localhost:50051')
stub = RPCExample_pb2_grpc.RPCExampleStub(channel)

response = stub.PredictFromInput(RPCExample_pb2.PredictFromInputRequest(input='test'))

print(f"client received: {response.input}")