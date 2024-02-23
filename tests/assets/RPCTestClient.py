import sys
import grpc
import RPCTest_pb2_grpc, RPCTest_pb2

channel = grpc.insecure_channel(f'localhost:8004')

stub = RPCTest_pb2_grpc.RPCTestStub(channel)

print(f"Response from TestRPCOneArgOneReturnRequest: {stub.TestRPCOneArgOneReturn(RPCTest_pb2.TestRPCOneArgOneReturnRequest(input0='test'))}")


print(f"Response from TestRPCTwoArgTwoReturnRequest: {stub.TestRPCTwoArgTwoReturn(RPCTest_pb2.TestRPCTwoArgTwoReturnRequest(input0='test0', input1=1))}")