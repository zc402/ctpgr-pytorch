from pathlib import Path
from constants.enum_keys import PG
from numpy import random
import numpy as np
from numpy.random import choice

from pgdataset.s1_skeleton import PgdSkeleton
from torch.utils.data import Dataset


# Randomly select and cut a video based on length. Longer videos are expected to be chosen more often.
class PgdTruncate(Dataset):
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        self.sk = PgdSkeleton(data_path, is_train, resize_img_size)
        # List of num_frames in each video
        frames_per_video = np.array([self.sk[s][PG.NUM_FRAMES] for s in range(len(self.sk))])
        self.__total_frames = np.asscalar(np.sum(frames_per_video))  # Total frames in whole dataset
        # which video does a frame in total_frames belong to
        tidx2vidx = [np.full(shape=n, fill_value=e) for e, n in enumerate(frames_per_video)]
        self.__tidx2vidx = np.concatenate(tidx2vidx, axis=0)  # [0,0,1,1,1,2,2 ...]
        # which local frame (index inside a single video) dose a frame in total_frames belong to
        tidx2lidx = [np.arange(n, dtype=np.int) for n in frames_per_video]
        self.__tidx2lidx = np.concatenate(tidx2lidx, axis=0)  # [0,1,0,1,2,0,1 ...]

        self.clip_len = clip_len
        # 标签后移
        self.LABEL_DELAY = 15  # LABEL_DELAY frames are delayed to leave some time for RNN to observe the gesture

    def __len__(self):
        return self.__total_frames

    def __getitem__(self, index):
        # TODO: 使用准确的Video Frames不使用random了
        vidx = self.__tidx2vidx[index]
        lidx = self.__tidx2lidx[index]
        res_dict = self.sk[vidx]

        self.__check_clip_len(res_dict)

        truncate = slice(lidx, lidx + self.clip_len)
        res_dict[PG.COORD_NATIVE] = self.__extend(res_dict[PG.COORD_NATIVE])[truncate]
        res_dict[PG.COORD_NORM] = self.__extend(res_dict[PG.COORD_NORM])[truncate]
        res_dict[PG.GESTURE_LABEL] = self.__extend(res_dict[PG.GESTURE_LABEL])[truncate]
        # 标签后移
        res_dict[PG.GESTURE_LABEL] = np.concatenate((np.zeros(self.LABEL_DELAY, dtype=np.int), res_dict[PG.GESTURE_LABEL]), axis=0)[:self.clip_len]
        return res_dict

    def __check_clip_len(self, res_dict):

        if self.clip_len == -1:  # -1: Use full video
            raise NotImplementedError("Deprecated")

        v_len = len(res_dict[PG.GESTURE_LABEL])
        if v_len <= self.clip_len:
            raise ValueError("Video %s too short (%d) for clip_len %d" %
                             (res_dict[PG.VIDEO_PATH], v_len, self.clip_len))

    @ staticmethod
    def __extend(feature):
        """Extent features to prevent indexOutOfBound when lidx+clip_len > video num_frames.
        This solution may impact performance"""
        return np.concatenate([feature, feature], axis=0)