from pathlib import Path
from .s1_temporal_coord_dataset import TemporalCoordDataset
from .s2_random_clip_dataset import RandomClipDataset
from pgdataset.bone_length_angle.bone_length_angle import BoneLengthAngle
from constants.enum_keys import PG
from torch.utils.data import Dataset

# 输出骨骼长度和角度特征的数据集
class LenAngDataset(Dataset):
    """Return handcrafted features: bone length and angle"""
    def __init__(self, data_path, is_train, clip_len):
        self.coord_ds = TemporalCoordDataset(data_path, is_train)
        self.rand_clip = RandomClipDataset(self.coord_ds, clip_len)
        self.bone_len_ang = BoneLengthAngle()

    def __len__(self):
        return len(self.rand_clip)

    def __getitem__(self, t_index):
        # index 是所有视频总帧数之中的某一帧
        res_dict = self.rand_clip[t_index]
        # PG.COORD_NORM: numpy array of shape: (F, X, J). F: frames, X: xy(2), K: keypoints
        feature_dict = self.bone_len_ang.parse(res_dict[PG.COORD_NORM])
        res_dict.update(feature_dict)
        return res_dict  # Use PG.ALL_HANDCRAFTED as features

