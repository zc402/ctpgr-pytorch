"""
ST-GCN
输入: N,C,T,V
输出：Class Score
"""
from pathlib import Path
from st_gcn.st_gcn_no_fc import StgcnNoFc
from torch.nn import LSTM
from torch import nn
import torch


class GCN_LSTM(nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        self.num_out_stgcn = 256  # num_output of STGCN
        self.num_out_lstm = 48
        self.num_classes = 9
        self.batch_size = batch_size

        self.ckpt_path = Path("checkpoints/gcn_lstm.pt")
        self.gcn = StgcnNoFc(2, self.num_out_stgcn)
        self.lstm = LSTM(input_size=self.num_out_stgcn, hidden_size=self.num_out_lstm)
        self.lin1 = nn.Linear(self.num_out_lstm, self.num_classes)
        self.drop = nn.Dropout(p=0.5)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)

    def save_ckpt(self):
        torch.save(self.state_dict(), self.ckpt_path)
        print('st-gcn model ckpt saved.')

    def load_ckpt(self, allow_new=True):
        if Path.is_file(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.load_state_dict(checkpoint)
            print('st-gcn model ckpt loaded.')
        else:
            if allow_new:
                print('new st-gcn model ckpt created.')
            else:
                raise FileNotFoundError('st-gcn model ckpt not found.')

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