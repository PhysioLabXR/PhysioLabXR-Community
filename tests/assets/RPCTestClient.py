import sys
import grpc
import RPCTest_pb2_grpc, RPCTest_pb2

channel = grpc.insecure_channel(f'localhost:8004')

stub = RPCTest_pb2_grpc.RPCTestStub(channel)
response = stub.TestRPCOneArgOneReturn(RPCTest_pb2.TestRPCOneArgOneReturnRequest(input0='test'))