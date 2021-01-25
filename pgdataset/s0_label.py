from typing import List
import csv
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from constants.enum_keys import PG


class PgdLabel(Dataset):

    def __init__(self, data_path, is_train):
        self.is_train = is_train

        if is_train:
            root: Path = data_path / "train"
        else:
            root: Path = data_path / "test"

        if not root.exists():
            raise FileNotFoundError(str(root), ' not found.')

        video_paths: List = list(root.glob('./*.mp4'))

        csv_paths: List = [p.with_suffix('.csv') for p in video_paths]
        csv_contents: List = [self.__load_csv_label(p) for p in csv_paths]

        self.video_csv = list(zip(video_paths, csv_contents))

    def __len__(self):
        return len(self.video_csv)

    def __getitem__(self, index):
        v_path, label = self.video_csv[index]
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
