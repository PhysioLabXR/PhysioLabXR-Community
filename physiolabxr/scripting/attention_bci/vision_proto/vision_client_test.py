import grpc

from physiolabxr.scripting.attention_bci.vision_proto import vision_pb2_grpc, vision_pb2

channel = grpc.insecure_channel('127.0.0.1:5555')
stub = vision_pb2_grpc.VisionStub(channel)

response = stub.PushInt(vision_pb2.PushIntRequest(value=1))

print(f"{response}")