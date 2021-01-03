import argparse
from pathlib import Path
import basic_tests.basic_tests
import pgdataset.s1_skeleton_coords
import train.train_police_gesture_model
import train.train_keypoint_model
import pred.play_keypoint_results
import pred.play_gesture_results
import pred.prepare_skeleton_from_video


def prepare_skeleton():
    pred.prepare_skeleton_from_video.save()


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--train_keypoint', action='store_true',
                    help="Train human keypoint estimation model from ai_challenger dataset")
parser.add_argument('-g', '--train_gesture', action='store_true',
                    help='Train police gesture recognition model from police gesture dataset')
parser.add_argument('-c', '--clean_saved_skeleton', action='store_true',
                    help='Delete saved skeleton from generated/coords to regenerate them during next training')

parser.add_argument('-a', '--play_keypoint', type=int,
                    help='Play keypoint estimation result')
parser.add_argument('-b', '--play_gesture', type=int,
                    help='Play gesture recognition result')
parser.add_argument('-p', '--play', type=str,
                    help='Assign a custom video path to play and recognize police gestures')

parser.add_argument('-u', '--unit_test', action='store_true',
                    help='Run unit test')

# play_custom_video('C:/Users/zc/PoliceGestureLong/test/012.mp4')

args = parser.parse_args()
if args.train_keypoint:
    train.train_keypoint_model.Trainer(batch_size=10, debug_mode=False).train()
elif args.train_gesture:
    train.train_police_gesture_model.Trainer().train()
elif args.clean_saved_skeleton:
    pgdataset.s1_skeleton_coords.SkeletonCoordsDataset.remove_generated_skeletons()
elif args.play_keypoint is not None:
    prepare_skeleton()
    pred.play_keypoint_results.Player().play(is_train=False, video_index=args.play_keypoint)
elif args.play_gesture is not None:
    prepare_skeleton()
    pred.play_gesture_results.Player().play_dataset_video(is_train=False, video_index=args.play_gesture)
elif args.play is not None:
    video_path = args.play
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path, ' is not a file')
    pred.play_gesture_results.Player().play_custom_video(video_path)
elif args.unit_test:
    basic_tests.basic_tests.run_tests()
