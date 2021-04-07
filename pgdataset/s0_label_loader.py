from typing import List
import csv
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from constants.enum_keys import PG


class LabelLoader():
    """Load .csv label and .mp4 video path"""

    def __init__(self, data_path, is_train):
        """label_root: folder of where .mp4 and .csv files are placed"""

        if is_train:
            label_root = data_path / "train"
        else:
            label_root = data_path / "test"

        if not label_root.exists():
            raise FileNotFoundError(str(label_root), ' not found.')

        video_paths: List = list(label_root.glob('./*.mp4'))

        csv_paths: List = [p.with_suffix('.csv') for p in video_paths]
        csv_contents: List = [self.__load_csv_label(p) for p in csv_paths]

        self.__video_csv = list(zip(video_paths, csv_contents))

    def num_videos(self) -> int:
        """Number of video files"""
        return len(self.__video_csv)

    def num_frames_per_video(self) -> np.ndarray:
        """array of shape [frames]. used for clipping."""
        frames_per_video = []
        for s in range(self.num_videos()):
            _, label = self.__video_csv[s]
            num_frames = len(label)
            frames_per_video.append(num_frames)
        frames_per_video = np.array(frames_per_video)
        return frames_per_video

    def __getitem__(self, index):
        v_path, label = self.__video_csv[index]
        v_name = v_path.name
        v_path = str(v_path)
        label = [int(l) for l in label]
        label = np.asarray(label, dtype=np.int)
        num_frames = label.shape[0]
        return {PG.VIDEO_NAME: v_name, PG.VIDEO_PATH: v_path, PG.GESTURE_LABEL: label, PG.NUM_FRAMES: num_frames}

    @staticmethod
    def __load_csv_label(csv_path):
        """
        Load csv labels. Each number indicates a gesture in a frame.
        example content: 0,0,0,2,2,2,2,2,0,0,0,0,0
        """
        with open(csv_path, newline='') as csv_file:
            reader = csv.reader(csv_file)
            row0 = list(reader)[0]

        return row0
