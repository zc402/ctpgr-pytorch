from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from pgdataset.s3_length_angle_dataset import LenAngDataset
from constants.enum_keys import PG
from models.stgcn import STGModel
from torch import optim
from constants import settings
from pgdataset.s1_temporal_coord_dataset import TemporalCoordDataset
from pgdataset.s2_random_clip_dataset import RandomClipDataset



class Trainer:
    def __init__(self, is_unittest=False):
        self.is_unittest = is_unittest
        self.batch_size = 10  # Not bigger than num of training videos
        self.clip_len = 15*4  # 15 fps
        self.data_loader = self.build_data_loader(self.clip_len, self.batch_size)
        self.model = STGModel()
        self.model.train()
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def build_data_loader(self, clip_len, batch_size):
        coord = TemporalCoordDataset(Path.home() / 'PoliceGestureLong', True)
        clip = RandomClipDataset(coord, clip_len)
        data_loader = DataLoader(clip, batch_size=batch_size, shuffle=False, num_workers=settings.num_workers)
        return data_loader

    def train(self):
        step = 1
        self.model.load_ckpt()
        for epoch in range(100):
            for ges_data in self.data_loader:
                # Shape: (N,F,X,C) N:Batch F:Frame X:xy C:coord
                # In GCN paper, represented as: N,T,C,V. C:xy, V:joints
                features = ges_data[PG.COORD_NORM]
                # Expect: N,C,T,V
                features = features.permute(0, 2, 1, 3)
                features = features.to(self.model.device, dtype=torch.float32)

                class_out = self.model(features)
                target = ges_data[PG.GESTURE_LABEL]  # N, F
                target = target[:, self.clip_len // 2]  # 取中间帧作为目标
                target = target.to(self.model.device, dtype=torch.long)

                # Cross Entropy, Input: (N, C), Target: (N).
                loss_tensor = self.loss(class_out, target)
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()

                if step % 100 == 0:
                    print("Step: %d, Loss: %f" % (step, loss_tensor.item()))
                if step % 5000 == 0:
                    self.model.save_ckpt()
                if self.is_unittest:
                    break
                step = step + 1
            if self.is_unittest:
                break
