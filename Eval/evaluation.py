from pathlib import Path

from itertools import groupby

from Eval.Matrics.edit_distance import EditDistance
from constants.enum_keys import PG
from pgdataset.s0_label_loader import LabelLoader
from pred.play_gesture_results import Player
from pgdataset.s1_temporal_coord_dataset import TemporalCoordDataset
from pgdataset.s2_random_clip_dataset import RandomClipDataset

class Eval:
    def __init__(self):
        self.player = Player()
        self.ed = EditDistance()
        self.data_path = Path.home() / 'PoliceGestureLong'

    def edit_distance(self):
        num_video = LabelLoader(self.data_path, is_train=False).num_videos()
        for n in range(num_video):
            res = self.player.play_dataset_video(is_train=False, video_index=n, show=False)
            target = res[PG.GESTURE_LABEL]
            source = res[PG.PRED_GESTURES]
            assert len(source) == len(target)
            source_group = [k for k, g in groupby(source)]
            target_group = [k for k, g in groupby(target)]
            S, D, I = self.ed.edit_distance(source_group, target_group)
            print('S:%d, D:%d, I:%d'%(S, D, I))
            pass

    def mean_jaccard_index(self):
        coord_ds = TemporalCoordDataset(self.data_path, is_train=False)
        clip_ds = RandomClipDataset(coord_ds, 15*5)  # "random" is caused by torch dataloader. Without dataloader the data will be generated in sequence.
        for clip in clip_ds:
            # clip shape: (T, C, V)
            pass

