from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from pgdataset.s3_handcraft import PgdHandcraft
from constants.enum_keys import PG
from models.gesture_recognition_model import GestureRecognitionModel
from torch import optim
from constants import settings


class Trainer:
    def __init__(self, is_unittest=False):
        self.is_unittest = is_unittest
        self.batch_size = 2  # Not bigger than num of training videos
        self.clip_len = 15*30
        pgd = PgdHandcraft(Path.home() / 'PoliceGestureLong', True, (512, 512), clip_len=self.clip_len)
        self.data_loader = DataLoader(pgd, batch_size=self.batch_size, shuffle=False, num_workers=settings.num_workers)
        self.model = GestureRecognitionModel(batch=self.batch_size)
        self.model.train()
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        step = 1
        self.model.load_ckpt()
        for epoch in range(100):
            for ges_data in self.data_loader:
                # Shape: (N,F,C) N:Batch F:Frame C:Channel(concatenated features)
                features = torch.cat((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                                      ges_data[PG.BONE_ANGLE_SIN]), dim=2)
                features = features.permute(1, 0, 2)  # NFC->FNC
                features = features.to(self.model.device, dtype=torch.float32)
                h0, c0 = self.model.h0(), self.model.c0()
                # class_out: (batch, num_class)
                _, h, c, class_out = self.model(features, h0, c0)
                target = ges_data[PG.GESTURE_LABEL]
                target = target.to(self.model.device, dtype=torch.long)
                target = target.permute(1, 0)
                # Cross Entropy, Input: (N, C), Target: (N).
                target = target.reshape((-1))  # new shape: (seq_len*batch)
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
