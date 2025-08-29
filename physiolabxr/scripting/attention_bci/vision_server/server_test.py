#----------- the backend to process the images captured by Fixation Camera and Wingman Camera

#----- gRPC service -------
import asyncio
import grpc
from physiolabxr.scripting.attention_bci.vision_proto import vision_pb2, vision_pb2_grpc
from google.protobuf import empty_pb2

class VisionService(vision_pb2_grpc.VisionServicer):
    """
    This is the class definition for vision service,
    which should include the track of current state index,
    save and analyze the streamed image information
    """

    async def Analyze(self, request, context):
        """

        Args:
            request:
                takes all the request from the image capturing
            context:
            the grpc channel context

        Returns:
            should process and return the target object IDs

        """

        # ---- unpack request ----
        ts   = request.timestamp_ms
        img_color  = bytes(request.color_image)  # your colorData
        img_depth  = bytes(request.depth_image)  # your depthData
        bbox_json = request.bbox_json
        gazed_item_index_dtn = bytes(request.gaze_item_index_dtn)
        current_screen_location_bytes = bytes(request.current_screen_location_bytes)

        # ---- analyze received infor ----


        # Test result
        result_dict = {0:0.8, 1:0.98, 2: 0.7}
        print("Returning the result:", result_dict)

        return vision_pb2.AnalyzeReply(id_scores=result_dict)

    async def PushInt(self, request, context):
        """

        Args:
            request:
                the integer value to keep track of the current state index
            context:
            the grpc channel context

        Returns:
            No return value, just trying to communicate the unity with the remote rpc server

        """
        value = request.value
        # Log / enqueue / update state, etc.
        print(f"PushInt received: {value}")
        return empty_pb2.Empty()


async def serve():
    # Increase message limits if you’ll send big images
    server = grpc.aio.server()
    vision_pb2_grpc.add_VisionServicer_to_server(VisionService(), server)

    # Listen only on localhost (127.0.0.1) port 5555
    port_fixation = server.add_insecure_port("127.0.0.1:5550")
    port_wingman = server.add_insecure_port("127.0.0.1:5551")
    port_state_index = server.add_insecure_port("127.0.0.1:5552")

    await server.start()
    print("gRPC Python aio server listening starts")
    await server.wait_for_termination()


if __name__ == "__main__":
    """
    should be the function call that dealing with RLPF logic calls
    """

    asyncio.run(serve())

    # TODO: update the out dated socket binds
    # the socket CONFLATEs are all 1's
    # fixation_cam_socket = get_cam_socket("tcp://127.0.0.1:5556", 'ColorDepthCamGazePositionBBox')
    # wingman_cam_socket = get_cam_socket("tcp://127.0.0.1:5557", 'ColorDepthCamGazePositionBBox')
    #
    # ovtr_fixation_socket = get_cam_socket("tcp://127.0.0.1:5560", 'OVTRCamFixation')
    # ovtr_wingman_socket = get_cam_socket("tcp://127.0.0.1:5551", 'OVTRCamWingman')


