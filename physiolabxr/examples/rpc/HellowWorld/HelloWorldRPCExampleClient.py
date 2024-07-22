import grpc
from physiolabxr.examples.rpc.HellowWorld import HelloWorldRPCExample_pb2_grpc, HelloWorldRPCExample_pb2

channel = grpc.insecure_channel('192.168.1.11:8004')
stub = HelloWorldRPCExample_pb2_grpc.HelloWorldRPCStub(channel)

response = stub.SayHello(HelloWorldRPCExample_pb2.SayHelloRequest(name='python client'))

print(f"{response.message}")