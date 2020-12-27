from pathlib import Path
from constants.enum_keys import PG
from numpy import random

from pgdataset.s1_skeleton_coords import SkeletonCoordsDataset

class TruncateDataset(SkeletonCoordsDataset):
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        super().__init__(data_path, is_train, resize_img_size)
        self.clip_len = clip_len

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        v_len = len(res_dict[PG.GESTURE_LABEL])
        if v_len <= self.clip_len:
            raise ValueError("Video %s too short (%d) for clip_len %d" %
                             (res_dict[PG.VIDEO_PATH], v_len, self.clip_len))
        start = random.randint(v_len-self.clip_len)
        truncate = slice(start, start + self.clip_len)
        res_dict[PG.COORD_NATIVE] = res_dict[PG.COORD_NATIVE][truncate]
        res_dict[PG.COORD_NORM] = res_dict[PG.COORD_NORM][truncate]
        res_dict[PG.GESTURE_LABEL] = res_dict[PG.GESTURE_LABEL][truncate]
        return res_dict
