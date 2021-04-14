"""
ST-GCN
输入: N,C,T,V
输出：Class Score
"""
from pathlib import Path
from st_gcn.st_gcn_no_fc import StgcnNoFc
from torch.nn import LSTM
from models import BaseModel
from torch import nn
import torch


class STGCN_LSTM(BaseModel):
    def __init__(self, batch_size):
        super().__init__()

        self.num_out_stgcn = 256  # num_output of STGCN
        self.num_out_lstm = 48
        self.num_classes = 9
        self.batch_size = batch_size

        self.gcn = StgcnNoFc(2, self.num_out_stgcn)
        self.lstm = LSTM(input_size=self.num_out_stgcn, hidden_size=self.num_out_lstm)
        self.lin1 = nn.Linear(self.num_out_lstm, self.num_classes)
        self.drop = nn.Dropout(p=0.5)

        self._to_device()

    def _get_model_name(self) -> str:
        return "GCN-LSTM"

    def forward(self, x, h, c):
        gcn_out = self.gcn(x)  # N, F, 256
        gcn_out = gcn_out.permute(1, 0, 2)  # NFC->FNC

        lstm_out, (hn, cn) = self.lstm(gcn_out, (h, c))
        class_out = self.lin1(lstm_out.view(-1, self.num_out_lstm))
        class_out = self.drop(class_out)
        return hn, cn, class_out

    def h0(self):
        return torch.randn((1, self.batch_size, self.num_out_lstm), device=self.device)

    def c0(self):
        return torch.randn((1, self.batch_size, self.num_out_lstm), device=self.device)