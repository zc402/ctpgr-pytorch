from pathlib import Path

from itertools import groupby

import torch
import numpy as np
from eval.Matrics.edit_distance import EditDistance
from constants.enum_keys import PG
from models.stgcn import STGModel
from pgdataset.s0_label_loader import LabelLoader
from pred.play_gesture_results import Player
from pgdataset.s1_temporal_coord_dataset import TemporalCoordDataset
from pgdataset.s2_random_clip_dataset import RandomClipDataset
from sklearn.metrics import jaccard_score

class Eval:
    def __init__(self):
        self.ed = EditDistance()
        self.data_path = Path.home() / 'PoliceGestureLong'

    def edit_distance(self):
        player = Player()
        num_video = LabelLoader(self.data_path, is_train=False).num_videos()
        for n in range(num_video):
            res = player.play_dataset_video(is_train=False, video_index=n, show=False)
            target = res[PG.GESTURE_LABEL]
            source = res[PG.PRED_GESTURES]
            assert len(source) == len(target)
            source_group = [k for k, g in groupby(source)]
            target_group = [k for k, g in groupby(target)]
            S, D, I = self.ed.edit_distance(source_group, target_group)
            print('S:%d, D:%d, I:%d'%(S, D, I))
            pass

    def mean_jaccard_index_gcn(self):
        clip_len = 15*4
        coord_ds = TemporalCoordDataset(self.data_path, is_train=False)
        clip_ds = RandomClipDataset(coord_ds, clip_len)  # "random" is caused by torch dataloader. Without dataloader the data will be generated in sequence.

        pred_list = []
        gt_list = []
        model = STGModel()
        model = model.eval()
        model.load_ckpt()

        for i, clip in enumerate(clip_ds):
            # clip shape: (T, C, V)
            target = clip[PG.GESTURE_LABEL]  # F
            target = target[clip_len // 2]  # 取中间帧作为目标

            features = clip[PG.COORD_NORM]
            features = torch.from_numpy(features)[None]  # Add new dimension
            features = features.permute(0, 2, 1, 3)
            features = features.to(model.device, dtype=torch.float32)
            with torch.no_grad():
                class_out = model(features)  # Expect input: N,C,T,V  # Out: N, class
            class_out = class_out.cpu()
            class_n = np.argmax(class_out, axis=1)
            pred_list.append(class_n.item())
            gt_list.append(target)

            if i > 15*100:
                break

        print(gt_list)
        print(pred_list)
        js = jaccard_score(gt_list, pred_list, average='micro')
        print("jaccard score:", round(js * 100, 2) )

    def mean_jaccard_index_gcn_lstm(self):
        pass

