from pathlib import Path

from itertools import groupby

import torch
import numpy as np
from eval.Matrics.edit_distance import EditDistance
from constants.enum_keys import PG
from models.stgcn import GCN_FC
from models.stgcn_lstm import GCN_LSTM
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

        model = GCN_FC()
        model = model.eval()
        model.load_ckpt()

        for video_dict in coord_ds:
            start = 20
            end = 15*10

            features = video_dict[PG.COORD_NORM][start: end]  # T,C,V (frame,xy,spatial)
            features = np.transpose(features, axes=[1, 0, 2])  # CTV
            features = features[np.newaxis]  # CTV -> NCTV
            features = torch.from_numpy(features)
            features = features.to(model.device, dtype=torch.float32)
            with torch.no_grad():
                # class_out: N*T, C
                class_out = model(features)

            gt_label = video_dict[PG.GESTURE_LABEL][start:end]  # T

            class_out = class_out.cpu()
            pred = np.argmax(class_out, axis=1)  # T

            print(gt_label)
            print(pred)
            js = jaccard_score(gt_label, pred, average='micro')
            print("jaccard score:", round(js * 100, 2))


    def mean_jaccard_index_gcn_lstm(self):
        coord_ds = TemporalCoordDataset(self.data_path, is_train=False)

        model = GCN_LSTM(batch_size=1)
        model = model.eval()
        model.load_ckpt()

        for video_dict in coord_ds:
            # res_dict 包含一个视频的全部标签等数据：
            # {PG.VIDEO_NAME, PG.VIDEO_PATH, PG.GESTURE_LABEL, PG.NUM_FRAMES}
            # {PG.COORD_NATIVE, PG.COORD_NORM}
            start = 20
            end = start + 15 * 100
            h, c = model.h0(), model.c0()

            features = video_dict[PG.COORD_NORM][start: end]  # T,C,V (frame,xy,spatial)
            features = np.transpose(features, axes=[1, 0, 2])  # CTV
            features = features[np.newaxis]  # CTV -> NCTV
            features = torch.from_numpy(features)
            features = features.to(model.device, dtype=torch.float32)
            with torch.no_grad():
                # class_out: N*T, C
                _, _, class_out = model(features, h, c)

            gt_label = video_dict[PG.GESTURE_LABEL][start: end]  # T

            class_n = np.argmax(class_out.cpu(), axis=1)

            print(gt_label)
            print(class_n)
            js = jaccard_score(gt_label, class_n, average='micro')
            print("jaccard score:", round(js * 100, 2))


