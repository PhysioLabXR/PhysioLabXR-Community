import sys
import grpc
import RPCTest_pb2_grpc, RPCTest_pb2
from google.protobuf import empty_pb2

channel = grpc.insecure_channel(f'localhost:8004')

stub = RPCTest_pb2_grpc.RPCTestStub(channel)

print(f"Response from TestRPCOneArgOneReturnRequest: {stub.TestRPCOneArgOneReturn(RPCTest_pb2.TestRPCOneArgOneReturnRequest(input0='test'))}")

print(f"Response from TestRPCTwoArgTwoReturnRequest: {stub.TestRPCTwoArgTwoReturn(RPCTest_pb2.TestRPCTwoArgTwoReturnRequest(input0='test0', input1=1))}")

print(f"Response from TestRPCNoInputNoReturn: {stub.TestRPCNoInputNoReturn(empty_pb2.Empty())}")

print(f"Response from TestRPCNoReturn: {stub.TestRPCNoReturn(RPCTest_pb2.TestRPCNoReturnRequest(input0=1))}")

print(f"Response from TestRPCNoArgs: {stub.TestRPCNoArgs(empty_pb2.Empty())}")


