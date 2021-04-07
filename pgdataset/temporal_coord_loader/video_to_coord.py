"""Parse human keypoint from a video frame by frame"""
from pred.human_keypoint_pred import HumanKeypointPredict
import cv2
import numpy as np
from constants.enum_keys import PG


class VideoToCoord:

    def __init__(self, video_folder, resize_img_size):
        # Lazy initialize keypoint prediction model
        # because the predictor is used only when there is no cache on disk
        self.predictor = None
        self.video_folder = video_folder
        self.resize_img_size = resize_img_size

    def predict(self, video_name):
        if self.predictor is None:
            self.predictor = HumanKeypointPredict()
        v_path = self.video_folder / video_name

        v_reader = self.__video_reader(v_path)
        native_list = []  # shape: (num_frames, xy(2), num_keypoints)
        norm_list = []  # shape: (num_frames, xy(2), num_keypoints)
        for i, frame in enumerate(v_reader):
            coord_dict = self.predictor.get_coordinates(frame)
            native_list.append(coord_dict[PG.COORD_NATIVE])
            norm_list.append(coord_dict[PG.COORD_NORM])
            print('Predicting %s: %d' % (video_name, i))
        native_list = np.asarray(native_list)
        norm_list = np.asarray(norm_list)
        return {PG.COORD_NATIVE: native_list, PG.COORD_NORM: norm_list}

    def __video_reader(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # Checking
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened" % video_path)
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if v_fps != 15:
            raise ValueError("video %s must have a frame rate of 15, currently %d" % (video_path, v_fps))

        # Read frames
        for _ in range(v_size):
            ret, img = cap.read()
            re_img = cv2.resize(img, self.resize_img_size)
            yield re_img

        cap.release()
        print("Video %s prediction finished" % video_path)
