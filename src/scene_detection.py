import os
import shutil
import numpy as np
import cv2
from typing import List
from PIL import Image
import pandas as pd
from collections import OrderedDict
import random
# from scenedetect import scene_manager as scene_manager_writer
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

from sklearn.metrics.pairwise import cosine_similarity

from src.image_flow import ImageFlow
from src.embedder import Embedder

# video path: path to mp4
# frame_paths: path to frames
# fps: frames per second (-> cv2)
# min_py_scenes: threshold to decide whether we need motion-based scene detection -> use as microservice parameter
# detector_type: scenedetect variable -> use as microservice parameter for possible new releases of scenedetect (for now we dont change defaults)
# content_detectoin: scenedetect variable  -> use as microservice parameter for possible new releases of scenedetect
class SceneDetector():
    def __init__(self, video_path,
                 frame_paths: List[str],
                 fps: int,
                 debug_handler,
                 min_py_scenes: int = 2,
                 detector_type: str = "content",
                 content_detection_threshold: int = 30,
                 num_images_per_scene: int = 3,
                 validate_frames=False,
                 apply_motion_filter=True,
                 apply_blur_filter=False,
                 verbose=True):

        self.debug_handler = debug_handler
        self.verbose = verbose
        self.frame_paths = frame_paths
        self.fps = fps
        self.validate_frames = validate_frames
        self.video_manager = VideoManager([video_path])
        # Set downscale factor to improve processing speed.
        self.video_manager.set_downscale_factor()
        self.stats_manager = StatsManager()
        # Construct our SceneManager and pass it our StatsManager.
        self.scene_manager = SceneManager(self.stats_manager)
        self.num_images_per_scene = num_images_per_scene
        if detector_type == "content":
            self.scene_manager.add_detector(ContentDetector())
            self.content_detection_threshold = content_detection_threshold

        self.min_py_scenes = min_py_scenes  # minimum number of scenes we expect from py scenedetect before resorting to motion-based
        self.min_motion_threshold = 0.05
        self.min_scene_duration = 0.0
        self.visual_similarity_threshold_motion: float = 0.99
        self.visual_similarity_threshold_pyscene: float = 0.98
        self.has_motion = False
        self.motion_delta = 0
        self.step_size = 5
        self.motion_frame_paths = frame_paths[::self.step_size]
        self.apply_motion_filter = apply_motion_filter
        self.apply_blur_filter = apply_blur_filter
        self.BLUR_THRESHOLD = 50

    def detect_scenes(self, n_scene_frames: int, embedder: Embedder):
        scene_df = pd.DataFrame()
        image_flow = ImageFlow()
        scene_dict = self.extract_pyscenes(n_scene_frames)

        if len(scene_dict) >= self.min_py_scenes:  # extract final scenes via scenedetect

            # scene_dict = self._check_last_frame(scene_dict, image_flow, img2vec)
            scene_df = self.get_pyscenes(scene_dict, image_flow, embedder, post_process=True)

            if self.verbose:
                self.debug_handler.show_scene_df(scene_df, scene_type="py-scene-detect")
        else:  # use motion-based
            self.validate_frames = False
            scene_df = self.get_motion_scenes(image_flow, embedder)
            self.has_motion = True
            if self.verbose:
                self.debug_handler.show_scene_df(scene_df, scene_type="motion-based")

        scene_df["blur"] = scene_df["scene_image_path"].apply(
            lambda f: cv2.Laplacian(cv2.imread(f, 0), cv2.CV_64F).var())
        if self.apply_blur_filter:
            # st.write("apply blur filter")
            scene_df = scene_df.query(f"blur > {self.BLUR_THRESHOLD}")

        scene_df.reset_index(drop=True, inplace=True)
        scene_df["frame_number"] = scene_df.index.tolist()

        return scene_df

    def get_pyscenes(self, scene_dict, image_flow, embedder: Embedder, post_process=True) -> pd.DataFrame:

        if post_process:
            # st.write("postprocess pyscenes")
            delete_idx = []
            for scene in scene_dict.keys():
                frame_paths = self._post_process_scenes(scene, scene_dict[scene]["frame_paths"], image_flow, embedder)
                scene_dict[scene]["frame_paths"] = frame_paths
                if len(frame_paths) == 0:
                    delete_idx.append(scene)

            for i in delete_idx:
                scene_dict.pop(i)

            if len(delete_idx) > 0:
                scene_dict = {i + 1: v for i, v in zip(np.arange(len(scene_dict)), list(scene_dict.values()))}

        scene_numbers = []
        frame_numbers = []
        frame_similarities = []
        frame_paths = []
        frame_durations = []
        start_frames = []
        end_frames = []
        start_times = []
        end_times = []
        durations = []
        scene_cnt = 0
        for i, scene in enumerate(scene_dict.keys()):
            scene_frames = scene_dict[scene]["frame_paths"]
            start_time = scene_dict[scene]["start_time"]
            duration = scene_dict[scene]["duration"]
            start_frame = scene_dict[scene]["start_frame"]
            end_frame = scene_dict[scene]["end_frame"]

            start_times += [start_time] * len(scene_frames)
            frame_durations += [duration] * len(scene_frames)
            end_times += [start_time + duration] * len(scene_frames)
            start_frames += [start_frame] * len(scene_frames)
            end_frames += [end_frame] * len(scene_frames)
            scene_numbers += [scene] * len(scene_frames)
            scene_vecs = [embedder.img2vec(Image.open(f)) for f in scene_frames]
            similarities = [cosine_similarity([scene_vecs[i]], [scene_vecs[i + 1]]) for i in
                            range(len(scene_frames) - 1)]
            similarities.append(-99)
            frame_similarities.extend(similarities)
            frame_paths.extend(scene_frames)

        # print(f'scene_number {len(scene_numbers)} scene_image_path {len(frame_paths)} length_frames: {len(frame_durations)} start_frame {len(start_frames)} end_frame  {len(end_frames)} start_time : {len(start_times)} end_time: {len(end_times)} duration: {len(durations)}')
        return pd.DataFrame(
            {"scene_number": scene_numbers, "scene_image_path": frame_paths, "length_frames": frame_durations,
             "start_frame": start_frames, "end_frame": end_frames, "start_time": start_times, "end_time": end_times,
             "duration": frame_durations, "similarity": frame_similarities})

    def extract_pyscenes(self, n_scene_frames: int, frame_padding = 2 ):
        #frame_padding how many frames to skip at boundaries (often black at beginning, blurry at transitions)
        try:
            # Start video_manager.
            self.video_manager.start()

            # Perform scene detection on video_manager.
            self.scene_manager.detect_scenes(frame_source=self.video_manager)

            # Obtain list of detected scenes.
            # Each scene is a tuple of (start, end) FrameTimecodes.
            scene_list = self.scene_manager.get_scene_list()

            start_times = [np.round(s[0].get_seconds(), 1) for s in scene_list]
            scene_durations = [np.round(s[1].get_seconds() - s[0].get_seconds(), 1) for s in scene_list]

            # WATCH-OUT - we use these scene_frame_IDs to access frame_paths array
            # This assumes every frame was extracted (since scenedetect accesses entire video to estimate scenes)
            # feel free to de-couple this, would be more flexible
            scene_frame_ID_lst = [(s[0].get_frames(), s[1].get_frames() - 1) for s in scene_list]
            scene_dict = {}
            scene_cnt = 0

            for scene_id, scene_frame_IDs in enumerate(scene_frame_ID_lst):
                # try:
                start_frame_ID = scene_frame_IDs[0] + frame_padding
                end_frame_ID = scene_frame_IDs[1] - frame_padding
                end_frame_ID = min(len(self.frame_paths) - 1, end_frame_ID)
                if start_frame_ID < end_frame_ID:
                    middle_frame_ID = start_frame_ID + (end_frame_ID - start_frame_ID) // 2
                    if len(scene_list) < 3 or (end_frame_ID - start_frame_ID) > 150:  # 5sec
                        # st.write(f" scene {scene_id} scene detection: taking 5 frames per scene")
                        frame_ID_first_half = start_frame_ID + (middle_frame_ID - start_frame_ID) // 2
                        frame_ID_second_half = middle_frame_ID + (end_frame_ID - middle_frame_ID) // 2
                        scene_frames = [self.frame_paths[start_frame_ID], self.frame_paths[frame_ID_first_half],
                                        self.frame_paths[middle_frame_ID], self.frame_paths[frame_ID_second_half],
                                        self.frame_paths[end_frame_ID]]
                    else:

                        if n_scene_frames == 1:
                            scene_frames = [self.frame_paths[middle_frame_ID]]
                        elif n_scene_frames == 2:
                            scene_frames = [self.frame_paths[start_frame_ID], self.frame_paths[end_frame_ID]]
                        elif n_scene_frames == 3:
                            scene_frames = [self.frame_paths[start_frame_ID], self.frame_paths[middle_frame_ID],
                                            self.frame_paths[end_frame_ID]]
                        else:
                            scene_frames = random.sample(self.frame_paths[start_frame_ID:end_frame_ID], n_scene_frames)
                    if self.validate_frames:
                        valid_frames = self._validate_frames(scene_frames)
                    else:
                        valid_frames = scene_frames
                    if len(valid_frames) > 0:  # and scene_durations[scene_id] >= self.min_scene_duration:
                        scene_cnt += 1
                        scene_dict[scene_cnt] = {}
                        scene_dict[scene_cnt]["start_time"] = start_times[scene_id]
                        scene_dict[scene_cnt]["duration"] = scene_durations[scene_id]
                        scene_dict[scene_cnt]["frame_paths"] = valid_frames
                        scene_dict[scene_cnt]["start_frame"] = int(
                            os.path.splitext(os.path.basename(scene_frames[0]))[0]) - frame_padding
                        scene_dict[scene_cnt]["end_frame"] = int(
                            os.path.splitext(os.path.basename(scene_frames[-1]))[0]) + frame_padding
            # except:
            #    pass

            scene_dict = {i: v for i, v in zip(np.arange(len(scene_dict)), list(scene_dict.values()))}
        finally:
            self.video_manager.release()

        return scene_dict

        ############# MOTION #######################

    def get_motion_scenes(self, image_flow, embedder: Embedder):
        scene_dict = self._compute_motion_scenes(image_flow)
        print("motion scenes", scene_dict)
        scene_dict = self._post_process_motion_scenes(scene_dict, image_flow, embedder)
        print("post proc motion scenes", scene_dict)
        scene_dict = self._check_last_frame(scene_dict, image_flow, embedder)
        print("_check_last_frame", scene_dict)
        #if len(scene_dict) == 1:
        scene_dict = self._check_first_frame(scene_dict, image_flow, embedder)
        print("_check_first_frame", scene_dict)

        frame_paths = [sd["frame_path"] for sd in scene_dict.values()]
        cum_scene_durations = [sd["duration_cumulative"] for sd in scene_dict.values()]
        scene_durations = [cum_scene_durations[0]] + list(np.diff(cum_scene_durations))
        frame_numbers = [sd["frame_number"] for sd in scene_dict.values()]
        frame_durations = [frame_numbers[0]] + list(np.diff(frame_numbers))

        start_times = []
        start_frames = []

        start_times.append(0)
        start_frames.append(1)

        for i in range(1, len(scene_dict)):
            start_times.append(cum_scene_durations[i] - scene_durations[i])
            start_frames.append(frame_numbers[i] - frame_durations[i])

        scene_df = pd.DataFrame(
            {"scene_number": list(scene_dict.keys()), "scene_image_path": frame_paths, "length_frames": frame_durations,
             "start_frame": start_frames, "end_frame": frame_numbers, "start_time": start_times,
             "end_time": cum_scene_durations, "duration": scene_durations})
        delete_scenes = []
        for idx, row in scene_df.iterrows():
            if row["duration"] < self.min_scene_duration:
                delete_scenes.append(idx)

            # if len(scene_df) - len(delete_scenes) == 1 and 0 in delete_scenes: #only one scene left -> keep first scene
            #    delete_scenes.remove(0)

        scene_df = scene_df[~scene_df["scene_number"].isin(delete_scenes)]
        scene_df.reset_index(drop=True, inplace=True)
        if len(scene_df) == 1:
            scene_df.loc[0, "scene_number"] = 0
            scene_df.loc[0, "start_frame"] = 1
            scene_df.loc[0, "end_time"] = scene_df.loc[0, "start_time"] + scene_df.loc[0, "end_time"]
            scene_df.loc[0, "start_time"] = 0
        print("scene_df", scene_df)
        return scene_df

    def _post_process_scenes(self, scene, frame_paths, image_flow, embedder: Embedder):
        delete_idx = []

        ## calculate motion in scene motions
        frame_imgs = [cv2.imread(f, 0) for f in frame_paths]
        frame_motions = [image_flow.get_motion(frame_imgs[i], frame_imgs[i + 1]) for i in range(len(frame_paths) - 1)]
        frame_motions = np.array(frame_motions)

        motion_idx = list(np.where(frame_motions < self.min_motion_threshold)[0])
        delete_idx.extend(motion_idx)

        scene_vecs = [embedder.img2vec(Image.open(f)) for f in frame_paths]
        similarities = [cosine_similarity([scene_vecs[i]], [scene_vecs[i + 1]]) for i in range(len(frame_paths) - 1)]
        similarities = np.array(similarities)
        similarity_idx = list(np.where(similarities > self.visual_similarity_threshold_pyscene)[0])
        delete_idx.extend(similarity_idx)
        delete_idx = np.unique(delete_idx)
        frame_paths = [f for i, f in enumerate(frame_paths) if i not in delete_idx]
        return frame_paths

    def _post_process_motion_scenes(self, scene_dict: dict, image_flow, embedder: Embedder):
        frame_paths = [sd["frame_path"] for sd in scene_dict.values()]
        scene_durations = [sd["duration"] for sd in scene_dict.values()]
        scene_durations = np.array(scene_durations)
        delete_idx = []
        delete_idx.extend(list(np.where(scene_durations < self.min_scene_duration)[0]))

        ## recalculate motion in scene motions
        if self.apply_motion_filter:
            frame_imgs = [cv2.imread(f, 0) for f in frame_paths]
            frame_motions = [image_flow.get_motion(frame_imgs[i], frame_imgs[i + 1]) for i in
                             range(len(frame_paths) - 1)]
            frame_motions = np.array(frame_motions)

            motion_idx = list(np.where(frame_motions < self.min_motion_threshold)[0])
            delete_idx.extend(motion_idx)
            print(delete_idx)

        scene_vecs = [embedder.img2vec(Image.open(f)) for f in frame_paths]

        similarities = [cosine_similarity([scene_vecs[i]], [scene_vecs[i + 1]]) for i in range(len(frame_paths) - 1)]
        similarities = np.array(similarities)
        similarity_idx = list(np.where(similarities > self.visual_similarity_threshold_motion)[0])
        # print(f" similarity: {similarity_idx}")
        delete_idx.extend(similarity_idx)
        # print(delete_idx)
        delete_idx = np.unique(delete_idx)
        # print(delete_idx)

        ## if only one scene left and it is the first or last frame -> keep it
        if len(delete_idx) > 0:
            if (len(scene_dict) - len(delete_idx)) == 1 and delete_idx[0] == 0:
                delete_idx = []

        if len(delete_idx) == len(scene_dict):
            delete_idx = delete_idx[:-1]  # keep last frame

        for idx in delete_idx:
            scene_dict.pop(idx)

        if len(delete_idx) > 0:
            # make sure scene numbers remain in +1 increments
            # getattr() hack to ensure we end up with python native ints (vs numpy.int64)
            scene_dict = OrderedDict({getattr(i, "tolist", lambda: i)(): v for i, v in
                          zip(np.arange(len(scene_dict)), list(scene_dict.values()))})

        return scene_dict

        # if there is only one "scene", check if we need to add the very first frame as additional "scene"

    def _check_first_frame(self, scene_dict: dict, image_flow, embedder: Embedder):
        # current first scene frame number
        current_first_frame_path = list(scene_dict.values())[0]["frame_path"]
        current_first_frame_number = list(scene_dict.values())[0]["frame_number"]

        real_first_frame_path = self.frame_paths[1]
        real_first_frame_number = int(os.path.splitext(os.path.basename(real_first_frame_path))[0])

        current_first = current_first_frame_number - real_first_frame_number
        if current_first > 1:
            real_first_frame_path = self.frame_paths[current_first//2]
            real_first_frame_number = int(os.path.splitext(os.path.basename(real_first_frame_path))[0])

            if self._is_valid_frame(real_first_frame_path):
                # enough motion?
                first_frame_motion = image_flow.get_motion(cv2.imread(current_first_frame_path, 0),
                                                           cv2.imread(real_first_frame_path, 0))
                if first_frame_motion > self.min_motion_threshold:
                        # dissimilar enough?
                    current_first_vec = embedder.img2vec(Image.open(current_first_frame_path))
                    real_first_vec = embedder.img2vec(Image.open(real_first_frame_path))
                    if cosine_similarity([current_first_vec], [real_first_vec]) < self.visual_similarity_threshold_motion:
                        num_scenes = len(scene_dict)
                        scene_dict[num_scenes] = {}
                        scene_dict[num_scenes]["frame_path"] = real_first_frame_path
                        scene_dict[num_scenes]["frame_number"] = real_first_frame_number
                        scene_dict[num_scenes]["duration_cumulative"] = real_first_frame_number / self.fps
                        scene_dict[num_scenes]["duration"] = real_first_frame_number / self.fps
                        scene_dict.move_to_end(num_scenes, last=False)

                        scene_dict = OrderedDict({i: v for i, v in zip(np.arange(len(scene_dict)), list(scene_dict.values()))})
        return scene_dict

    # given current last frame, check whether we need to add the very last frame as extra scene
    def _check_last_frame(self, scene_dict, image_flow, embedder: Embedder):
        # current last scene frame number
        current_last_frame_path = list(scene_dict.values())[-1]["frame_path"]
        current_last_frame_number = list(scene_dict.values())[-1]["frame_number"]

        # last video frame number
        real_last_frame_path = self.frame_paths[-2]  # second last, very last often different/irrelevant
        real_last_frame_number = int(os.path.splitext(os.path.basename(real_last_frame_path))[0])

        if real_last_frame_number > current_last_frame_number and self._is_valid_frame(real_last_frame_path):
            # get duration
            last_frame_duration_cumulative = real_last_frame_number / self.fps
            last_frame_duration = last_frame_duration_cumulative - current_last_frame_number / self.fps
            # last "scene" long enough?
            if last_frame_duration > self.min_scene_duration:
                # enough motion?
                last_frame_motion = image_flow.get_motion(cv2.imread(current_last_frame_path, 0),
                                                          cv2.imread(real_last_frame_path, 0))
                if last_frame_motion > self.min_motion_threshold:
                    # dissimilar enough?
                    current_last_vec = embedder.img2vec(Image.open(current_last_frame_path))
                    real_last_vec = embedder.img2vec(Image.open(real_last_frame_path))

                    if cosine_similarity([current_last_vec],[real_last_vec]) < self.visual_similarity_threshold_motion:
                        current_last_scene_number = list(scene_dict.keys())[-1]
                        new_last_scene_number = current_last_scene_number + 1
                        scene_dict[new_last_scene_number] = OrderedDict()
                        scene_dict[new_last_scene_number]["frame_path"] = real_last_frame_path
                        scene_dict[new_last_scene_number]["frame_number"] = real_last_frame_number
                        scene_dict[new_last_scene_number]["duration_cumulative"] = real_last_frame_number / self.fps
                        scene_dict[new_last_scene_number]["duration"] = scene_dict[new_last_scene_number][
                                                                            "duration_cumulative"] - current_last_frame_number / self.fps

        return scene_dict

    def _compute_motion_scenes(self, image_flow):

        frame_motions, motion_median, motion_delta, num_motion_frames, motion_type = self._compute_motion_metrics(
            image_flow, self.motion_frame_paths, self.min_motion_threshold)

        print(f"motion delta {motion_delta} motion median {motion_median}")
        self.frame_motions = frame_motions
        self.motion_type = motion_type

        local_scene_frames = []
        scene_dict = OrderedDict()
        scene_cnt = 0
        for frame_path, frame_motion in zip(self.motion_frame_paths, frame_motions):
            if frame_motion > motion_median + motion_delta and frame_motion > self.min_motion_threshold/2:
                #local_scene_frames = [lsf for lsf in local_scene_frames if not lsf in list(scene_dict.values())]
                if len(local_scene_frames) > 0:
                    if self.validate_frames:
                        valid_frames = self._validate_frames(local_scene_frames)
                    else:
                        valid_frames = local_scene_frames

                    if len(valid_frames) > 0:
                        scene_dict[scene_cnt] = {}

                        scene_image = valid_frames[len(valid_frames) // 2]
                        scene_dict[scene_cnt]["frame_path"] = scene_image

                        frame_number = int(os.path.splitext(os.path.basename(scene_image))[0])
                        scene_dict[scene_cnt]["frame_number"] = frame_number

                        local_scene_frames = []
                        scene_cnt += 1

            else:
                local_scene_frames.append(frame_path)

        #local_scene_frames = [lsf for lsf in local_scene_frames if not lsf in list(scene_dict.values())]
        if len(local_scene_frames) > 0:
            if self.validate_frames:
                valid_frames = self._validate_frames(local_scene_frames)
            else:
                valid_frames = local_scene_frames
            if len(valid_frames) > 0:
                scene_dict[scene_cnt] = {}

                scene_image = valid_frames[len(valid_frames) // 2]
                scene_dict[scene_cnt]["frame_path"] = scene_image

                frame_number = int(os.path.splitext(os.path.basename(scene_image))[0])
                scene_dict[scene_cnt]["frame_number"] = frame_number

                local_scene_frames = []
                scene_cnt += 1

        frame_numbers = [sd["frame_number"] for sd in scene_dict.values()]
        frame_numbers = np.array(frame_numbers) / self.fps
        scene_durations = np.array([frame_numbers[0]] + list(np.diff(frame_numbers)))
        for i in range(scene_cnt):
            scene_dict[i]["duration"] = scene_durations[i]
            scene_dict[i]["duration_cumulative"] = frame_numbers[i]

        return scene_dict

    def _compute_motion_metrics(self, image_flow, frame_paths, min_motion_threshold):

        # frame_imgs = [image_resize(cv2.imread(f, 0), self.image_size_default) for f in frame_paths]
        frame_imgs = [cv2.imread(f, 0) for f in frame_paths]
        frame_motions = [image_flow.get_motion(frame_imgs[i], frame_imgs[i + 1]) for i in range(len(frame_paths) - 1)]

        frame_motions = np.array(frame_motions)

        motion_mean = np.mean(frame_motions)
        motion_median = np.median(
            frame_motions)  ### downstream used for: iMAT -> "Catches Attention" & iMAT -> "Connects"; as input to model for final KPI prediction
        print(motion_mean, motion_median)
        motion_std = np.std(frame_motions)

        local_motion_idcs = np.where(frame_motions > min_motion_threshold)[0]

        num_motion_frames = len(
            local_motion_idcs)  ### iMAT -> Catches Attention ; as input to model for final KPI prediction

        motion_type = self._get_motion_label(motion_mean, num_motion_frames, len(frame_paths), min_motion_threshold)

        motion_delta = 0
        if motion_type == "continuous motion" or motion_type == "high motion":
            motion_delta = motion_std / 2

        return frame_motions, motion_median, motion_delta, num_motion_frames, motion_type

    @staticmethod
    def _get_motion_label(motion_mean, num_motion_frames, num_frames, min_motion_threshold):
        if num_motion_frames == 0:
            return "no motion"
        elif num_motion_frames > 0 and num_motion_frames >= num_frames - 4:
            return "continuous motion"
        elif motion_mean > min_motion_threshold and motion_mean < 1:
            return "medium motion"
        elif motion_mean >= 1:
            return "high motion"
        elif num_motion_frames > 0 and motion_mean < min_motion_threshold:
            return "minimal motion"

    def _validate_frames(self, local_scene_frames, save_images=True):
        valid_frames = []
        for f in local_scene_frames:
            if self._is_valid_frame(f):
                valid_frames.append(f)

        return valid_frames

    @staticmethod
    def _is_valid_frame(frame):
        im = Image.open(frame)
        extrema = im.convert("L").getextrema()
        if (extrema[0] < 5 and extrema[1] < 5) or (
                extrema[0] > 250 and extrema[1] > 250):  # avoid empty white or black frames
            return False
        else:
            return True

