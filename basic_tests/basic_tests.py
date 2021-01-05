import time
import unittest
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from aichallenger import AicNative, AicResize, AicAugment, AicGaussian, AicAffinityField, AicNorm
from pgdataset.s0_label_loader import LabelLoader
from pgdataset.s1_skeleton_coords import SkeletonCoordsDataset
from pgdataset.s3_handcrafted_features import HandCraftedFeaturesDataset
from pgdataset.s2_truncate import TruncateDataset

import pgdataset.s1_skeleton_coords
import train.train_police_gesture_model
import train.train_keypoint_model
import pred.play_keypoint_results
import pred.play_gesture_results
import pred.prepare_skeleton_from_video
import pred.evaluation


def dataset_next(dataset: Dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=lambda x: x)
    it = iter(loader)
    next(it)

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_hkd_s0(self):
        ds = AicNative(Path.home() / "AI_challenger_keypoint", is_train=True)
        dataset_next(ds)

    def test_hkd_s1(self):
        ds = AicResize(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512))
        dataset_next(ds)

    def test_hkd_s2(self):
        ds = AicAugment(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512))
        dataset_next(ds)

    def test_hkd_s3(self):
        ds = AicGaussian(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512), heat_size=(64, 64))
        dataset_next(ds)

    def test_hkd_s4(self):
        ds = AicAffinityField(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512), heat_size=(64, 64))
        dataset_next(ds)

    def test_hkd_s5(self):
        ds = AicNorm(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512), heat_size=(64, 64))
        dataset_next(ds)

    def test_pgd_s3(self):
        ds = HandCraftedFeaturesDataset(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512), clip_len=15*3)
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=1, collate_fn=lambda x: x)
        it = iter(loader)
        next(it)

    # def test_keypoint_training(self):
    #     train.train_keypoint_model.Trainer(batch_size=2, debug_mode=False, is_unittest=True).train()


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataset)
    unittest.TextTestRunner(verbosity=0).run(suite)

