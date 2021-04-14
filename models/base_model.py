from pathlib import Path
from st_gcn.st_gcn_fc import StgcnFc
from torch import nn
import torch

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.__ckpt_path = Path('checkpoints') / self._get_model_name()
        self.__ckpt_path = self.__ckpt_path.with_suffix(".ckpt")
        self.__model_name = self._get_model_name()

    def _to_device(self):
        """应当在变量初始化完成后再调用"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)

    def _get_model_name(self) -> str:
        raise NotImplementedError()

    def save_ckpt(self):
        torch.save(self.state_dict(), self.__ckpt_path)
        print(self.__model_name, ': checkpoint saved.')

    def load_ckpt(self, allow_new=True):
        if Path.is_file(self.__ckpt_path):
            checkpoint = torch.load(self.__ckpt_path)
            self.load_state_dict(checkpoint)
            print(self.__model_name, ': checkpoint loaded.')
        else:
            if allow_new:
                print(self.__model_name, ': new checkpoint created.')
            else:
                raise FileNotFoundError(self.__model_name, ': checkpoint not found.')
