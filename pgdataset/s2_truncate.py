from pathlib import Path
from constants.enum_keys import PG
from numpy import random
import numpy as np

from pgdataset.s1_skeleton_coords import SkeletonCoordsDataset

class TruncateDataset(SkeletonCoordsDataset):
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        super().__init__(data_path, is_train, resize_img_size)
        self.clip_len = clip_len
        self.LABEL_DELAY = 15  # LABEL_DELAY frames are delayed to leave some time for RNN to observe the gesture

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        if self.clip_len == -1:  # -1: Use full video
            return res_dict
        v_len = len(res_dict[PG.GESTURE_LABEL])
        if v_len <= self.clip_len:
            raise ValueError("Video %s too short (%d) for clip_len %d" %
                             (res_dict[PG.VIDEO_PATH], v_len, self.clip_len))
        start = random.randint(v_len-self.clip_len)
        truncate = slice(start, start + self.clip_len)
        res_dict[PG.COORD_NATIVE] = res_dict[PG.COORD_NATIVE][truncate]
        res_dict[PG.COORD_NORM] = res_dict[PG.COORD_NORM][truncate]
        res_dict[PG.GESTURE_LABEL] = res_dict[PG.GESTURE_LABEL][truncate]
        # 标签后移
        res_dict[PG.GESTURE_LABEL] = np.concatenate((np.zeros(self.LABEL_DELAY, dtype=np.int), res_dict[PG.GESTURE_LABEL]), axis=0)[:self.clip_len]
        return res_dict
