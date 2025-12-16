#----------- the backend to process the images captured by Fixation Camera and Wingman Camera

#----- gRPC service -------
import asyncio
import traceback

import grpc
from wingman.vision_server_rpc import vision_pb2, vision_pb2_grpc
from google.protobuf import empty_pb2
import lpips
from wingman.camera_utils import camera_capture_utils
from scripting.eyetracking.configs import *
from wingman.vision_data_config import *
from collections import defaultdict
# from supervised_client import *

class VisionService(vision_pb2_grpc.VisionServicer):
    """
    This is the class definition for vision service,
    which should include the track of current state index,
    save and analyze the streamed image information
    """

    """
        This is the class definition for vision service,
        which should include the track of current state index,
        save and analyze the streamed image information
        """

    def __init__(self):
        # the state lock to avoid rush condition
        self.lock = asyncio.Lock()

        # state index to keep track of the current game state
        self.state_index = 0

        # item index + is gazed to track whether the item is gazed or not
        self.item_gazed_state = {}

        self.train_data: TrainData = defaultdict(list)
        self.infer_data: InferenceData = defaultdict(list)

        # fix detection parameters  #######################################
        self.loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        self.previous_img_patch = None
        self.fixation_frame_counter = 0
        self.fixation_above_threshold = False

        self.wingman_addr = "127.0.0.1:50051"
        self.train_task = None
        self.model_trained = False

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
        try:
            camera, *rest = camera_capture_utils.unpack_analyze_request(request)

            if camera == "wingman":
                color_img, depth_img, timestamp, item_bboxes = rest

                # crop first (no lock)
                updates = []  # list[(item_index, cropped_image)]
                for item_index, item_bbox in item_bboxes.items():

                    if len(item_bbox) != 4:
                        continue

                    crop_rtn = camera_capture_utils.crop_bbox(color_img, *list(item_bbox))
                    if crop_rtn is None:
                        continue
                    cropped, item_rect = crop_rtn
                    if cropped.size > 0 and cropped.shape[0] >= MIN_CROP_SIZE and cropped.shape[1] >= MIN_CROP_SIZE:
                        updates.append((item_index, cropped))
                # commit updates (short lock)
                async with self.lock:
                    for idx, img in updates:
                        self.infer_data[idx].append(img)
            else:
                # fixation camera
                color_img, depth_img, timestamp, item_bboxes, gazed_item_index, gazed_item_dtn, gaze_x, gaze_y = rest

                '''
                # compute patch (no lock)
                img_patch_x_min = int(np.min([np.max([0, gaze_x - patch_size[0] / 2]), image_size[0] - patch_size[0]]))
                img_patch_x_max = int(np.max([np.min([image_size[0], gaze_x + patch_size[0] / 2]), patch_size[0]]))
                img_patch_y_min = int(np.min([np.max([0, gaze_y - patch_size[1] / 2]), image_size[1] - patch_size[1]]))
                img_patch_y_max = int(np.max([np.min([image_size[1], gaze_y + patch_size[1] / 2]), patch_size[1]]))
                img_patch = color_img[img_patch_x_min:img_patch_x_max, img_patch_y_min:img_patch_y_max]

                img_patch_rect = (img_patch_x_min, img_patch_y_min, img_patch_x_max, img_patch_y_max)

                # Take a quick snapshot of previous patch under lock,
                # but do the expensive similarity outside.
                async with self.lock:
                    prev_patch = self.previous_img_patch

                if prev_patch is not None:
                    img_tensor = camera_capture_utils.prepare_image_for_sim_score(img_patch)
                    prev_tensor = camera_capture_utils.prepare_image_for_sim_score(prev_patch)
                    distance = self.loss_fn_alex(img_tensor, prev_tensor).item()
                    fixation = 0 if distance > similarity_threshold else 1
                else:
                    fixation = 0
                '''

                if gazed_item_index in self.item_gazed_state:
                    fixation = self.item_gazed_state[gazed_item_index]
                else:
                    fixation = 0

                # Update fixation counters (short lock)
                async with self.lock:
                    if fixation == 0:
                        self.fixation_frame_counter = 0
                    else:
                        self.fixation_frame_counter += 1
                    self.fixation_above_threshold = self.fixation_frame_counter >= fixation_min_frame_count
                    # print(f"Fixation frame count: {self.fixation_frame_counter}, above threshold: {self.fixation_above_threshold}")
                    # self.previous_img_patch = img_patch

                # Prepare batch updates without holding the lock
                train_adds = []  # list[((item_index, dtn), cropped_image)]
                infer_adds = []  # list[(item_index, cropped_image)]
                img_w_bbox = camera_capture_utils.plot_items_on_image(color_img, item_bboxes)
                cv2.imshow("vision server", img_w_bbox)
                for item_index, data in item_bboxes.items():
                    item_bbox = data["Item1"]
                    item_name = data["Item2"]
                    item_dtn = data["Item3"]

                    if len(item_bbox) != 4:
                        continue

                    crop_rtn = camera_capture_utils.crop_bbox(color_img, *list(item_bbox))
                    if crop_rtn is None:
                        continue
                    cropped, item_rect = crop_rtn
                    if cropped.size == 0 or cropped.shape[0] < MIN_CROP_SIZE or cropped.shape[1] < MIN_CROP_SIZE:
                        continue

                    # Fixation detection using patch similarity - disabled
                    '''
                    # if camera_capture_utils.has_overlap(item_rect, img_patch_rect) and self.fixation_above_threshold:
                    '''

                    if self.fixation_above_threshold:
                        key = (int(item_index), int(item_dtn))  # Use item_index, item_dtn as key
                        train_adds.append((key, cropped))
                        print(f"Added 1 training image for item {item_index} dtn {item_dtn}")
                    else:
                        infer_adds.append((item_index, cropped))  # For inference, only use item_index (not dtn)

                # Commit updates (short lock)
                async with self.lock:
                    for key, img in train_adds:
                        self.train_data[key].append(img)
                    for idx, img in infer_adds:
                        self.infer_data[idx].append(img)

            # FEATURE: Training Gate (take a snapshot under lock)

            # print the current data status

            print("train data: ", {k: len(v) for k, v in self.train_data.items()})
            print("infer data:  ", {k: len(v) for k, v in self.infer_data.items()})

            async with self.lock:
                # quick snapshot to iterate outside
                non_ids = {idx for (idx, dtn), imgs in self.train_data.items() if int(dtn) == 1 and len(imgs) > 10}
                tar_ids = {idx for (idx, dtn), imgs in self.train_data.items() if int(dtn) == 2 and len(imgs) > 10}
                can_train = bool(non_ids) and bool(tar_ids) and (len(non_ids | tar_ids) >= 2) and len(tar_ids) >= 1

                # Make a shallow copy for the training task to read outside the lock
                train_snapshot = None
                if can_train and (self.train_task is None or self.train_task.done()):
                    train_snapshot = {k: list(v) for k, v in self.train_data.items()}

            if train_snapshot is not None:
                # FEATURE: Kick off non-blocking training if allowed
                print("Dispatching training task...")
                self.train_task = asyncio.create_task(_run_train(self.wingman_addr, train_snapshot))
                self.model_trained = True

            async with self.lock:
                infer_snapshot = {idx: list(imgs) for idx, imgs in self.infer_data.items() if len(imgs) > 0}

                # clear the queue to avoid repeatedly score old frames
                self.infer_data = defaultdict(list)

            id_scores = {}
            if infer_snapshot:
                # FEATURE:  Inference Gate
                try:
                    if (self.model_trained == True):
                        id_scores = await run_wingman_inference_from_buffers(self.wingman_addr, infer_snapshot,
                                                                             aggregate="mean")
                except Exception as e:
                    print(f"[Wingman][inference] error: {e}")

            result_dict = {int(k): float(v) for k, v in id_scores.items()} or {}
            # print("Returning the result:", result_dict)
            return vision_pb2.AnalyzeReply(id_scores=result_dict)
        except Exception as e:
            traceback.print_exc()
            raise e

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


