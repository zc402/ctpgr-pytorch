import time
import unittest
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from aichallenger import AicNorm
from pgdataset.s3_length_angle_dataset import LenAngDataset
from constants import settings

import train.train_police_gesture_model
import train.train_keypoint_model
import pred.play_keypoint_results
import pred.play_gesture_results
import pred.prepare_skeleton_from_video


def dataset_next(dataset: Dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=settings.num_workers, collate_fn=lambda x: x)
    it = iter(loader)
    next(it)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_hkd(self):
        ds = AicNorm(Path.home() / "AI_challenger_keypoint", is_train=True, resize_img_size=(512, 512), heat_size=(64, 64))
        dataset_next(ds)

    def test_pgd(self):
        ds = LenAngDataset(Path.home() / 'PoliceGestureLong', is_train=True, resize_img_size=(512, 512), clip_len=15 * 3)
        for i in range(1000):
            dataset_next(ds)

    def test_keypoint_training(self):
        # -k
        train.train_keypoint_model.Trainer(batch_size=2, is_unittest=True).train()

    def test_gesture_training(self):
        # -g
        train.train_keypoint_model.Trainer(batch_size=2, is_unittest=True).train()

    def test_play_keypoint(self):
        # -a 0
        pred.play_keypoint_results.Player(is_unittest=True).play(is_train=False, video_index=0)

    def test_play_gesture(self):
        # -b 0
        pred.play_gesture_results.Player(is_unittest=True).play_dataset_video(is_train=False, video_index=0)

    def test_play_gesture_custom(self):
        # -p C:\Users\?\PoliceGestureLong\test\012.mp4
        pred.play_gesture_results.Player(is_unittest=True).play_custom_video(Path.home()/'PoliceGestureLong'/'test'/'012.mp4')

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataset)
    unittest.TextTestRunner(verbosity=0).run(suite)

