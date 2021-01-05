# Predict each frame to coordinates and save to disk for further usages
from pred.human_keypoint_pred import HumanKeypointPredict
import pickle
import shutil
from pathlib import Path
import cv2
import numpy as np
from pgdataset.s0_label_loader import LabelLoader
from constants.enum_keys import PG


class SkeletonCoordsDataset(LabelLoader):
    """Load coords from disk if exists, else predict coords."""
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple):
        super().__init__(data_path, is_train)
        self.resize_img_size = resize_img_size
        if is_train:
            self.coord_folder = Path("generated/coords/train/")
            self.video_folder = data_path / "train"
        else:
            self.coord_folder = Path("generated/coords/test/")
            self.video_folder = data_path / "test"
        self.coord_folder.mkdir(parents=True, exist_ok=True)
        self.predictor = None  # Lazy initialize keypoint prediction model

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        v_name = res_dict[PG.VIDEO_NAME]
        coord_dict = self.__vpath_to_coords(v_name)
        # keys: 'coord_native', 'coord_norm'
        res_dict.update(coord_dict)
        return res_dict

    def __vpath_to_coords(self, video_name: str):
        coord_dict = self.__load_coords(video_name)
        if coord_dict is None:
            coord_dict = self.__predict_from_video(video_name)
            self.__save_coords(video_name, coord_dict)
        return coord_dict

    def __save_coords(self, video_name, coords):
        pkl_path = self.coord_folder / video_name
        pkl_path = pkl_path.with_suffix('.pkl')
        with pkl_path.open('wb') as pickle_file:
            pickle.dump(coords, pickle_file)

    def __load_coords(self, video_name):
        pkl_path = self.coord_folder / video_name
        pkl_path = pkl_path.with_suffix('.pkl')
        if not pkl_path.exists():
            return None
        with pkl_path.open('rb') as pickle_file:
            coords = pickle.load(pickle_file)
        return coords

    def __predict_from_video(self, video_name):
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

    @staticmethod
    def remove_generated_skeletons():
        p = Path("generated/coords/")
        shutil.rmtree(p)


