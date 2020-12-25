# Predict each frame to coordinates and save to disk for further usages
from pred.human_keypoint_pred import HumanKeypointPredict
import pickle
from pathlib import Path
import cv2
import numpy as np
from pgdataset.s0_label_loader import LabelLoader
from constants.enum_keys import PG

class SkeletonCoords(LabelLoader):
    """Load coords from disk if exists, else predict coords."""
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple):
        super().__init__(data_path, is_train)
        self.resize_img_size = resize_img_size
        if not is_train:
            raise NotImplementedError('the train skeletons are save in /generated/coords, logic of saving test skeletons not implemented')
        self.coord_folder = Path("../generated/coords/")
        self.video_folder = data_path / "train"
        self.coord_folder.mkdir(parents=True, exist_ok=True)
        self.predictor = None  # Lazy initialize keypoint prediction model

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        v_name = res_dict[PG.VIDEO_PATH].name
        coord_dict = self.__vpath_to_coords(v_name)
        # keys: 'coord_native', 'coord_norm'
        res_dict.update(coord_dict)
        return res_dict

    def __vpath_to_coords(self, video_name: str):
        coord_path = self.coord_folder / video_name
        if not coord_path.exists():
            coord_dict = self.__predict_from_video(video_name)
            self.__save_coords(video_name, coord_dict)
        coord_dict = self.__load_coords(video_name)
        return coord_dict

    def __save_coords(self, video_name, coords):
        pkl_path = self.coord_folder / video_name
        pkl_path = pkl_path.with_suffix('.pkl')
        pickle.dump(coords, pkl_path)

    def __load_coords(self, video_name):
        pkl_path = self.coord_folder / video_name
        pkl_path = pkl_path.with_suffix('.pkl')
        coords = pickle.load(pkl_path)
        return coords

    def __predict_from_video(self, video_name):
        if self.predictor is None:
            self.predictor = HumanKeypointPredict()
        v_path = self.video_folder / video_name

        v_reader = self.__video_reader(v_path)
        coord_list = []
        for i, frame in enumerate(v_reader):
            coord_dict = self.predictor.get_coordinates(frame)
            coord_list.append(coord_dict)
            print('Predicting:' + str(i))
        return coord_list

    def __video_reader(self, video_path):

        cap = cv2.VideoCapture(str(video_path))
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
            yield img

        cap.release()
        print("Video %s prediction finished" % video_path)



