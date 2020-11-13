from pathlib import Path

from torch.utils.data import DataLoader

from aichallenger.aic_resize import AicResize
from aichallenger.aic_augment import AicAugment
from aichallenger.aic_gaussian import AicGaussian
import cv2

def debug_resize():
    dataset = AicResize(Path.home() / "AI_challenger_keypoint", False, (512, 512))
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=True)
    for d in loader:
        print(d)
        break

def debug_aug():
    dataset = AicAugment(Path.home() / "AI_challenger_keypoint", True, (512, 512))
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=True)
    for d in loader:
        print(d)
        break

def debug_gau():
    dataset = AicGaussian(Path.home() / "AI_challenger_keypoint", True, (512, 512), (64, 64))
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=True)
    for d in loader:
        print(d)
        break

if __name__ == '__main__':
    debug_gau()
