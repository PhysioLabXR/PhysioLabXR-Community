#----------- the backend to process the images captured by Fixation Camera and Wingman Camera

#----- gRPC service -------
import asyncio
import traceback
from datetime import datetime
import time

import grpc
import pandas
from wingman.vision_server_rpc import vision_pb2, vision_pb2_grpc
from google.protobuf import empty_pb2
import lpips
from wingman.camera_utils import camera_capture_utils
from scripting.eyetracking.configs import *
from wingman.vision_data_config import *
from collections import defaultdict
from wingman.request_recorder import AsyncRequestRecorder
import os


# Minimum crop size to avoid degenerate images that cause PNG encoding/decoding issues
MIN_CROP_SIZE = 8

EXP_DF = pandas.DataFrame(columns=['block id', 'block difficulty', 'number of samples seen', 'seconds elapsed',
                                   'train precision', 'train recall', 'train accuracy','train loss',
                                   'wingman (val) precision', 'wingman (val) recall', 'wingman (val) accuracy','wingman (val) loss'])

base_dir = os.path.dirname(os.path.abspath(__file__))


class VisionService(vision_pb2_grpc.VisionServicer):
    """
    This is the class definition for vision service,
    which should include the track of current state index,
    save and analyze the streamed image information
    """
    def __init__(self):
        self.state_lock = asyncio.Lock()  # for state_index/difficulty
        self.data_lock = asyncio.Lock()  # for train_data/infer_data/etc

        # start time to keep track of the time elapsed
        self.start_time = time.time()

        # state index to keep track of the current game state
        self.state_index =0

        # difficulty level to keep track of the current game difficulty
        self.difficulty_level = 0

        # item index + is gazed to track whether the item is gazed or not
        self.item_gazed_state = {}

        self.train_data : TrainData = defaultdict(list)
        self.infer_data : InferenceData = defaultdict(list)

        # fix detection parameters  #######################################
        # self.loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        # self.previous_img_patch = None
        self.fixation_frame_counter = 0
        self.fixation_above_threshold = False

        self.wingman_addr = "localhost:50051"
        self.train_task = None
        self.model_trained = False

        # data collection
        # self.recorder = AsyncRequestRecorder(root_dir="data")
        # self.recorder.start()

        self.last_train_metrics = None
        self.last_record_metrics = None

        self.total_samples_seen = 0

        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = os.path.join(base_dir, f"wingman_log_{run_stamp}.csv")

        # recreate file + write header
        EXP_DF.iloc[0:0].to_csv(self.csv_path, index=False)

    def _on_train_done(self, task: asyncio.Task):
        try:
            metrics = task.result()
            if metrics:
                self.last_train_metrics = metrics
                print("[Wingman] Training completed:", metrics)
        except Exception as e:
            print("[Wingman] Training task error:", e)

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

                # COLLECT DATA FOR WINGMAN CAMERA
                self.recorder.enqueue(camera=camera, timestamp=timestamp, meta={"item_bboxes": item_bboxes}, color_img=color_img, depth_img=depth_img)

                # crop first (no lock)
                updates = []  # list[(item_index, cropped_image)]
                for item_index, item_bbox in item_bboxes.items():

                    if len(item_bbox)!=4:
                        continue

                    crop_rtn = camera_capture_utils.crop_bbox(color_img, *list(item_bbox))
                    if crop_rtn is None:
                        continue
                    cropped, item_rect = crop_rtn
                    if cropped.size > 0 and cropped.shape[0] >= MIN_CROP_SIZE and cropped.shape[1] >= MIN_CROP_SIZE:
                        updates.append((item_index, cropped))
                # commit updates (short lock)
                async with self.data_lock:
                    for idx, img in updates:
                        self.infer_data[idx].append(img)
            else:
                # fixation camera
                color_img, depth_img, timestamp, item_bboxes, gazed_item_index, gazed_item_dtn, gaze_x, gaze_y = rest

                # COLLECT DATA FOR FIXATION CAMERA
                # self.recorder.enqueue(camera=camera, timestamp=timestamp, meta={"item_bboxes": item_bboxes, "gazed_item_index": gazed_item_index, "gazed_item_dtn": gazed_item_dtn, "gaze_x": gaze_x, "gaze_y": gaze_y}, color_img=color_img, depth_img=depth_img)

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
                async with self.data_lock:
                    prev_patch = self.previous_img_patch

                if prev_patch is not None:
                    img_tensor = camera_capture_utils.prepare_image_for_sim_score(img_patch)
                    prev_tensor = camera_capture_utils.prepare_image_for_sim_score(prev_patch)
                    distance = self.loss_fn_alex(img_tensor, prev_tensor).item()
                    fixation = 0 if distance > similarity_threshold else 1
                else:
                    fixation = 0
                '''

                async with self.data_lock:
                    gazed_state = dict(self.item_gazed_state)  # snapshot
                # use gazed_state from here on

                if gazed_item_index in gazed_state:
                    fixation = gazed_state[gazed_item_index]
                else:
                    fixation = 0

                # print(f"Gazed item {gazed_item_index} dtn {gazed_item_dtn} fixation: {fixation}")

                # Update fixation counters (short lock)
                async with self.data_lock:
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

                ''' 
                # Debug visualization - disabled
                img_w_bbox = camera_capture_utils.plot_items_on_image(color_img, item_bboxes)
                cv2.imshow("vision server", img_w_bbox)
                key = cv2.waitKey(1) & 0xFF  # pumps events
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("vision server window closed.")
                '''

                for item_index, data in item_bboxes.items():
                    item_bbox = data["Item1"]
                    item_name = data["Item2"]
                    item_dtn = data["Item3"]

                    if len(item_bbox)!=4:
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
                        # print(f"Added 1 training image for item {item_index} dtn {item_dtn}")
                    else:
                        infer_adds.append((item_index, cropped))  # For inference, only use item_index (not dtn)

                # Commit updates (short lock)
                async with self.data_lock:
                    if train_adds:
                        for key, img in train_adds:
                            self.train_data[key].append(img)
                    if infer_adds:
                        for idx, img in infer_adds:
                            self.infer_data[idx].append(img)

            # FEATURE: Training Gate (take a snapshot under lock)

            # print the current data status

            print("train data: ", {k: len(v) for k, v in self.train_data.items()})
            print("infer data: ", {k: len(v) for k, v in self.infer_data.items()})
            # print("Fixation frame counter:", self.fixation_frame_counter, "Above threshold:", self.fixation_above_threshold)

            async with self.data_lock:
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
                self.train_task.add_done_callback(self._on_train_done)
                self.model_trained = True

            ''' 
            ---------------------------------
            Log training metrics if available  
            ---------------------------------  
            '''
            if hasattr(self, "last_train_metrics") and self.last_train_metrics is not None:
                # last training metrics available
                metrics = self.last_train_metrics
                if metrics != self.last_record_metrics:
                    self.total_samples_seen += metrics.get('samples_seen', 0)

                    EXP_DF.loc[len(EXP_DF)] = [
                        self.state_index,
                        self.difficulty_level,
                        self.total_samples_seen,
                        time.time() - self.start_time,
                        metrics.get('train_precision', 0.0),
                        metrics.get('train_recall', 0.0),
                        metrics.get('train_accuracy', 0.0),
                        metrics.get('train_loss', 0.0),
                        -1,
                        -1,
                        -1,
                        -1
                    ]

                    EXP_DF.tail(1).to_csv(
                        self.csv_path,
                        mode="a",
                        header=not os.path.exists(self.csv_path),
                        index=False
                    )
                    self.last_record_metrics = metrics

            async with self.data_lock:
                infer_snapshot = {idx: list(imgs) for idx, imgs in self.infer_data.items() if len(imgs) > 0}

                # clear the queue to avoid repeatedly score old frames
                self.infer_data = defaultdict(list)

            id_scores = {}
            if infer_snapshot:
                # FEATURE:  Inference Gate
                try:
                    if self.model_trained == True:
                        # id_scores = await run_wingman_inference_from_buffers(self.wingman_addr, infer_snapshot, aggregate="mean")
                        print("Enter training")
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
        This function keeps track of the state index from Unity,
        The state index will change upon new request

        Args:
            request:
                the integer value to keep track of the current state index
            context:
            the grpc channel context
        Returns:
            No return value, just trying to communicate the unity with the remote rpc server
        """
        async with self.state_lock:
            block_id, difficulty = _unpack_block_difficulty(request.value)
            if self.state_index!= block_id or self.difficulty_level != difficulty:

                self.train_data = defaultdict(list)
                self.infer_data =defaultdict(list)
                self.fixation_above_threshold =False
                self.train_task = None
                self.model_trained = False

                # reset total samples seen
                self.total_samples_seen = 0

                # reset the fixation frame counter
                self.fixation_frame_counter = 0

                # reset the time
                self.start_time = time.time()

                print("State index changed, reset train and infer data.")

            self.state_index = block_id
            self.difficulty_level = difficulty
            print(f"State Index: {self.state_index}, Difficulty Level: {self.difficulty_level}")
        return empty_pb2.Empty()

    async def GazeFrame(self, request, context):
        """
        This function keeps track of the gaze frame from Unity,
        The gaze frame will change upon new request

        Args:
            request:
                the integer value to keep track of the current gaze frame
            context:
            the grpc channel context
        Returns:
            No return value, just trying to communicate the unity with the remote rpc server
        """
        async with self.data_lock:
            self.item_gazed_state = request.id_gazed
            # print(f"GazeFrame received: {self.item_gazed_state}")
        return empty_pb2.Empty()


'''
    The function to unpack the state index and difficulty level from the request
'''
def _unpack_block_difficulty(packed: int):
    difficulty = packed &  0xFF
    block_id = (packed >> 8) & 0x00FFFFFF
    return block_id, difficulty


'''
    The async function to run the training process
'''
async def _run_train(wingman_addr, train_snapshot):
    try:
        return None
        # return await run_wingman_training_from_buffers(
        #     wingman_addr, train_snapshot,
        # )
    except Exception as e:
        print(f"[Wingman][train] error: {e}")
        print(traceback.format_exc())
        return None


async def serve():
    # Increase message limits if you’ll send big images
    server = grpc.aio.server()
    vision_pb2_grpc.add_VisionServicer_to_server(VisionService(), server)

    # Listen only on localhost (127.0.0.1) port 5555

    '''
    Localhost Test
    '''
    # port_fixation
    server.add_insecure_port("127.0.0.1:5550")
    # port_wingman
    server.add_insecure_port("127.0.0.1:5551")
    # port_state_index
    server.add_insecure_port("127.0.0.1:5552")
    # port_gaze_index
    server.add_insecure_port("127.0.0.1:5553")

    await server.start()
    print("gRPC Python aio server listening starts")
    await server.wait_for_termination()


if __name__ == "__main__":
    """
        RLPF server run in the main
    """
    asyncio.run(serve())
