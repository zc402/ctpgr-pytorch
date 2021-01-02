import torch
from torch import nn
from torch.nn import LSTM
from constants.keypoints import aic_bones, aic_bone_pairs
from pathlib import Path

class GestureRecognitionModel(nn.Module):
    def __init__(self, batch):
        super().__init__()
        num_input = len(aic_bones) + 2*len(aic_bone_pairs)
        self.num_hidden = 48
        self.num_output = 9
        self.batch = batch
        self.rnn = LSTM(input_size=num_input, hidden_size=self.num_hidden)
        self.lin1 = nn.Linear(self.num_hidden, self.num_output)
        self.drop = nn.Dropout(p=0.5)

        self.ckpt_path = Path('checkpoints/lstm.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)

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
        # output shape: (seq_len, batch, num_directions * hidden_size)
        lstm_out, (hn, cn) = self.rnn(x, (h, c))
        class_out = self.lin1(lstm_out.view(-1, self.num_hidden))
        class_out = self.drop(class_out)
        return lstm_out, hn, cn, class_out

    def h0(self):
        return torch.randn((1, self.batch, self.num_hidden), device=self.device)

    def c0(self):
        return torch.randn((1, self.batch, self.num_hidden), device=self.device)