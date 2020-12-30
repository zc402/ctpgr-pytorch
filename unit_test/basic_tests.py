import unittest
from pathlib import Path

from torch.utils.data import DataLoader
from aichallenger import AicNative, AicResize, AicAugment, AicGaussian, AicAffinityField, AicNorm
from pgdataset.s0_label_loader import LabelLoader
from pgdataset.s1_skeleton_coords import SkeletonCoordsDataset
from pgdataset.s3_handcrafted_features import HandCraftedFeaturesDataset
from pgdataset.s2_truncate import TruncateDataset

class TestDataset(unittest.TestCase):

    def test_hkd_s0(self):
        ds = AicNative(Path.home() / "AI_challenger_keypoint", is_train=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_hkd_s1(self):
        ds = AicResize(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_hkd_s2(self):
        ds = AicAugment(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_hkd_s3(self):
        ds = AicGaussian(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512),
                         heat_size=(64, 64))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_hkd_s4(self):
        ds = AicAffinityField(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512),
                         heat_size=(64, 64))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_hkd_s5(self):
        ds = AicNorm(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512),
                         heat_size=(64, 64))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_pgd_s0(self):
        ds = LabelLoader(Path.home() / 'PoliceGestureLong', is_train=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_pgd_s1(self):
        ds = SkeletonCoordsDataset(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)
        # for d in loader:
        #     pass

    def test_pgd_s2(self):
        ds = TruncateDataset(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512), clip_len=15*3)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    def test_pgd_s3(self):
        ds = HandCraftedFeaturesDataset(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512), clip_len=15*3)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

if __name__ == '__main__':
    unittest.main()