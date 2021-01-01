import torch
from torch import nn
from torch.nn import LSTM
from constants.keypoints import aic_bones, aic_bone_pairs
from pathlib import Path

class GestureRecognitionModel(nn.Module):
    def __init__(self, batch):
        super().__init__()
        num_input = len(aic_bones) + 2*len(aic_bone_pairs)
        self.num_output = 9
        self.batch = batch
        self.rnn = LSTM(input_size=num_input, hidden_size=self.num_output)
        self.ckpt_path = Path('checkpoints/lstm.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)
        pass

    def save_ckpt(self):
        torch.save(self.state_dict(), self.ckpt_path)
        print('LSTM checkpoint saved.')

    def load_ckpt(self, allow_new=True):
        if Path.is_file(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.load_state_dict(checkpoint)
        else:
            if allow_new:
                print('LSTM ckpt not found.')
            else:
                raise FileNotFoundError('LSTM ckpt not found.')

    def forward(self, x, h, c):
        output, (hn, cn) = self.rnn(x, (h, c))
        return output, hn, cn

    def h0(self):
        return torch.randn((1, self.batch, self.num_output), device=self.device)

    def c0(self):
        return torch.randn((1, self.batch, self.num_output), device=self.device)