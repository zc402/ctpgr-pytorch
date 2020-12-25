import unittest
from pathlib import Path

from torch.utils.data import DataLoader
from pgdataset.s0_label_loader import LabelLoader
from pgdataset.s1_skeleton_coords import SkeletonCoords
from pgdataset.s2_handcrafted_features import HandCraftedFeatures

class TestDataset(unittest.TestCase):

    def test_pgd_s0(self):
        ds = LabelLoader(Path.home() / 'PoliceGestureLong', is_train=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_pgd_s1(self):
        ds = SkeletonCoords(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_pgd_s2(self):
        ds = HandCraftedFeatures(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

if __name__ == '__main__':
    unittest.main()