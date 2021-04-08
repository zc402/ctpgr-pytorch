"""
ST-GCN
输入: N,C,T,V
输出：Class Score
"""
from pathlib import Path
from st_gcn.st_gcn_fc import StgcnFc
from torch import nn
import torch

class GCN_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.ckpt_path = Path("checkpoints/gcn_fc.pt")
        self.model_pose = StgcnFc(2, 9)
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

    def forward(self, x):
        return self.model_pose(x)