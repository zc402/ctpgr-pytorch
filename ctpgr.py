import unittest
import basic_tests.basic_tests
import pred.play_keypoint_results

import train.train_keypoint_model

def train_keypoint():
    train.train_keypoint_model.Trainer(batch_size=10, debug_mode=False).train()

def run_unittest():
    basic_tests.basic_tests.run_tests()

def play_keypoint_results():
    pred.play_keypoint_results.Player().play(is_train=True, video_index=0)

play_keypoint_results()