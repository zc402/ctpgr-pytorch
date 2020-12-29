from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from pgdataset.s3_handcrafted_features import HandCraftedFeaturesDataset
from constants.enum_keys import PG
from models.recognition_LSTM import GestureRecognitionModel
from torch import optim

class Trainer:
    def __init__(self):
        self.batch_size = 1
        self.clip_len = 15*30
        pgd = HandCraftedFeaturesDataset(Path.home() / 'PoliceGestureLong', True, (512, 512), clip_len=self.clip_len)
        self.data_loader = DataLoader(pgd, batch_size=self.batch_size, shuffle=False)#, collate_fn=lambda x: x)
        self.model = GestureRecognitionModel(batch=self.batch_size, clip_len=self.clip_len)
        self.model.train()
        self.loss = CrossEntropyLoss()  # The input is expected to contain raw, unnormalized scores for each class.
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        step = 0
        self.model.load_ckpt()
        for epoch in range(10000):
            for ges_data in self.data_loader:
                # Shape: (N,F,C) N:Batch F:Frame C:Channel(concatenated features)
                features = torch.cat((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                                      ges_data[PG.BONE_ANGLE_SIN]), dim=2)
                features = features.permute(1, 0, 2)  # NFC->FNC
                features = features.to(self.model.device, dtype=torch.float32)
                h0, c0 = self.model.h0, self.model.c0
                # output of shape (seq_len, batch, num_directions * hidden_size)
                output, h, c = self.model(features, h0, c0)
                target = ges_data[PG.GESTURE_LABEL]

                loss_tensor = self.loss(output, target)
                self.opt.zero_grad()
                loss_tensor.backward()
                self.opt.step()

                step = step + 1
                if step % 10 == 0:
                    print("Step: %d, Loss: %d" % (step, loss_tensor.item()))


if __name__ == '__main__':
    Trainer().train()