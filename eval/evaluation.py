from pathlib import Path

from itertools import groupby
import pandas
import torch
import numpy as np
from eval.Matrics.edit_distance import EditDistance
from constants.enum_keys import PG
from models.bla_lstm import BLA_LSTM
from models.stgcn_fc import STGCN_FC
from models.stgcn_lstm import STGCN_LSTM
from models.stgcn_fc import STGCN_FC
from pgdataset.s0_label_loader import LabelLoader
from pred.play_gesture_results import Player
from pgdataset.s1_temporal_coord_dataset import TemporalCoordDataset
from pgdataset.s2_random_clip_dataset import RandomClipDataset
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


class Eval:

    def __init__(self, model_name: str):
        # self.ed = EditDistance()
        self.data_path = Path.home() / 'PoliceGestureLong'
        self.model_name = model_name
        if model_name == "BLA_LSTM":
            # joint coord -> bone length angle -> lstm
            self.pred_model = BLA_LSTM(1)
        elif model_name == "STGCN_FC":
            # joint coord -> STGCN -> fully connected
            self.pred_model = STGCN_FC()
        elif model_name == "STGCN_LSTM":
            # joint coord -> STGCN -> LSTM
            self.pred_model = STGCN_LSTM(1)

    # def edit_distance(self):
    #     player = Player()
    #     num_video = LabelLoader(self.data_path, is_train=False).num_videos()
    #     for n in range(num_video):
    #         res = player.play_dataset_video(is_train=False, video_index=n, show=False)
    #         target = res[PG.GESTURE_LABEL]
    #         source = res[PG.PRED_GESTURES]
    #         assert len(source) == len(target)
    #         source_group = [k for k, g in groupby(source)]
    #         target_group = [k for k, g in groupby(target)]
    #         S, D, I = self.ed.edit_distance(source_group, target_group)
    #         print('S:%d, D:%d, I:%d'%(S, D, I))
    #         pass

    def mean_jaccard_index(self):
        with torch.no_grad():
            self.__mean_jaccard_index_nograd()

    def __mean_jaccard_index_nograd(self):
        coord_ds = TemporalCoordDataset(self.data_path, is_train=False)

        model = self.pred_model
        model = model.eval()
        model.load_ckpt()

        gt_all = []
        pred_all = []
        res_dict = {}  # {filename: jaccard_score}
        for video_dict in coord_ds:
            start = 0
            end = video_dict[PG.NUM_FRAMES]

            features = video_dict[PG.COORD_NORM][start: end]  # (frame, xy, joint)

            if self.model_name == "BLA_LSTM":
                features = self.pred_model.coord_to_bla(features)
                _, _, _, class_out = model(features, model.h0(), model.c0())
            elif self.model_name == "STGCN_FC":
                features = np.transpose(features, axes=[1, 0, 2])  # CTV
                features = features[np.newaxis]  # CTV -> NCTV
                features = torch.from_numpy(features).to(model.device, dtype=torch.float32)
                class_out = model(features)  # class_out: N*T, C
            elif self.model_name == "STGCN_LSTM":
                features = np.transpose(features, axes=[1, 0, 2])  # CTV
                features = features[np.newaxis]  # CTV -> NCTV
                features = torch.from_numpy(features).to(model.device, dtype=torch.float32)
                _, _, class_out = model(features, model.h0(), model.c0())
            else:
                raise NotImplementedError()

            class_out = class_out.cpu().numpy()
            pred = np.argmax(class_out, axis=1)  # T

            gt_label = video_dict[PG.GESTURE_LABEL][start:end]  # T

            js = jaccard_score(gt_label, pred, average='micro')
            print(video_dict[PG.VIDEO_NAME], "jaccard score:", round(js * 100, 2), "%")
            res_dict[video_dict[PG.VIDEO_NAME]] = js

            pred_all.extend(pred)
            gt_all.extend(gt_label)

        # Jaccard Score ------------------------------------
        # per video
        js = jaccard_score(gt_all, pred_all, average='micro')
        print("js of all videos:", round(js * 100, 2), "%")
        res_dict["ALL"] = js
        with open(Path("generated", "gesture_results") / Path(self.model_name).with_suffix(".pkl"), "wb") as f:
            pickle.dump(res_dict, f)

        # per class
        res_dict = {}
        js = jaccard_score(gt_all, pred_all, average=None)
        for ci, class_score in enumerate(js):
            res_dict["C"+str(ci)] = class_score
        with open(Path("generated", "gesture_results") / Path(self.model_name).with_suffix(".class.pkl"), "wb") as f:
            pickle.dump(res_dict, f)
        pass

        # Confusion Matrix
        cf_matrix = confusion_matrix(gt_all, pred_all)
        ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.1%', cmap='Blues')
        _ = ax.set(xlabel="Predicted", ylabel="True")
        plt.savefig("docs/cm.pdf")


