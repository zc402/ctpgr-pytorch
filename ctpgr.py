import unittest
import basic_tests.basic_tests
import pred.play_keypoint_results
import train.train_recognition_model
import train.train_keypoint_model
import pred.play_gesture_results

def train_keypoint():
    train.train_keypoint_model.Trainer(batch_size=10, debug_mode=False).train()

def run_unittest():
    basic_tests.basic_tests.run_tests()

def play_keypoint_results():
    pred.play_keypoint_results.Player().play(is_train=True, video_index=5)

def train_recognition_model():
    train.train_recognition_model.Trainer().train()

def play_gesture_res():
    pred.play_gesture_results.Player().play(is_train=True, video_index=3)

play_gesture_res()