import torch
from torch import nn
from torch.nn import LSTM
from constants.keypoints import aic_bones, aic_bone_pairs
from pathlib import Path

class GestureRecognitionModel(nn.Module):
    def __init__(self, batch, clip_len):
        super().__init__()
        num_input = len(aic_bones) + 2*len(aic_bone_pairs)
        num_output = 9
        self.rnn = LSTM(input_size=num_input, hidden_size=num_output)
        self.ckpt_path = Path('../checkpoints/lstm.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)
        self.h0 = torch.randn((1, batch, num_output), device=self.device)
        self.c0 = torch.randn((1, batch, num_output), device=self.device)
        pass

    def save_ckpt(self):
        torch.save(self.state_dict(), self.model_path)
        print('LSTM checkpoint saved.')

    def load_ckpt(self):
        if Path.is_file(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.model_pose.load_state_dict(checkpoint)
        else:
            print('LSTM ckpt not found.')

    def forward(self, x, h, c):
        output, (hn, cn) = self.rnn(x, (h, c))
        return output, hn, cn