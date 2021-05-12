import argparse
from pathlib import Path
import basic_tests.basic_tests
from pgdataset.temporal_coord_loader import TemporalCoordLoader
import train.train_bla_lstm
import train.train_keypoint_model
import pred.play_keypoint_results
import pred.play_gesture_results
import pred.prepare_skeleton_from_video

from train.train_gcn import GcnTrainer
from train.train_gcn_lstm import GcnLstmTrainer
from train.train_bla_lstm import Trainer as BlaLstmTrainer

from eval.evaluation import Eval

def prepare_skeleton():
    pred.prepare_skeleton_from_video.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_keypoint', action='store_true',
                        help="Train human keypoint estimation model from ai_challenger dataset")
    parser.add_argument('--clean_saved_skeleton', action='store_true',
                        help='Clean the saved skeleton from generated/coords to regenerate them during next training')

    parser.add_argument('--train_gcn_fc', action='store_true',
                        help='Train GCN_FC (proposed method)')
    parser.add_argument('--train_gcn_lstm', action='store_true',
                        help='Train GCN_LSTM (for ablation experiment)')
    parser.add_argument('--train_bla_lstm', action='store_true',
                        help='Train BLA_LSTM (baseline, for ablation)')

    parser.add_argument('--eval_gcn_fc', action='store_true',
                        help='Compute Jaccard score of GCN_FC (proposed method)')
    parser.add_argument('--eval_gcn_lstm', action='store_true',
                        help='Compute Jaccard score of GCN_LSTM (for ablation experiment)')
    parser.add_argument('--eval_bla_lstm', action='store_true',
                        help='Compute Jaccard score of BLA_LSTM (baseline, for ablation)')

    args = parser.parse_args()
    # Train keypoint extractor
    if args.train_keypoint:
        train.train_keypoint_model.Trainer(batch_size=10).train()
    elif args.clean_saved_skeleton:
        TemporalCoordLoader.remove_generated_skeletons()

    # Train gesture recognizer
    elif args.train_gcn_fc:
        GcnTrainer().train()
    elif args.train_gcn_lstm:
        GcnLstmTrainer().train()
    elif args.train_bla_lstm:
        BlaLstmTrainer().train()

    # Evaluation
    elif args.eval_gcn_fc:
        Eval("STGCN_FC").mean_jaccard_index()
    elif args.eval_gcn_lstm:
        Eval("STGCN_LSTM").mean_jaccard_index()
    elif args.eval_bla_lstm:
        Eval("STGCN_LSTM").mean_jaccard_index()


