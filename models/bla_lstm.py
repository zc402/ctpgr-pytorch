import torch
from torch import nn
from torch.nn import LSTM

from constants.enum_keys import PG
from constants.keypoints import aic_bones, aic_bone_pairs
import numpy as np
from models import BaseModel
from pgdataset.bone_length_angle.bone_length_angle import BoneLengthAngle

class BLA_LSTM(BaseModel):
    def __init__(self, batch):
        super().__init__()
        self.bla = BoneLengthAngle()

        num_input = len(aic_bones) + 2*len(aic_bone_pairs)
        self.num_hidden = 48
        self.num_output = 9
        self.batch = batch
        self.lstm = LSTM(input_size=num_input, hidden_size=self.num_hidden)
        self.lin1 = nn.Linear(self.num_hidden, self.num_output)
        self.drop = nn.Dropout(p=0.5)

        self._to_device()

    def _get_model_name(self) -> str:
        return "BLA-LSTM"

    def coord_to_bla(self, coord_norm: np.ndarray):
        # Coordinates to Bone Length Angle
        # convert [TC] numpy to [TNC] pytorch
        ges_data = self.bla.parse(coord_norm)
        features = ges_data[PG.ALL_HANDCRAFTED]  # TC
        features = features[:, np.newaxis, :]  # TNC
        features = torch.from_numpy(features).to(self.device, dtype=torch.float32)
        return features  # Used as "x" in forward()

    def forward(self, x, h, c):
        # lstm input shape: (seq_len, batch, channels)
        # lstm output shape: (seq_len, batch, num_directions * hidden_size)
        lstm_out, (hn, cn) = self.lstm(x, (h, c))
        class_out = self.lin1(lstm_out.view(-1, self.num_hidden))
        class_out = self.drop(class_out)
        return lstm_out, hn, cn, class_out

    def h0(self):
        return torch.randn((1, self.batch, self.num_hidden), device=self.device)

    def c0(self):
        return torch.randn((1, self.batch, self.num_hidden), device=self.device)