# Continuous Traffic Police Gestures Recognizer Based on Spatial-Temporal Graph Convolution

## Download Trained model

Download from github release (v1.0.1):

https://github.com/zc402/ctpgr-pytorch/releases/

The "code-model-results.zip" file contains all trained models and evaluation materials and results.

## Download dataset (only for training)

### Keypoint
To train the CPM, please download AI Challenger dataset (~20GB):

https://arxiv.org/abs/1711.06475

(The official site is temporarily unavailable, but I can not publish it due to its license. 
Contact me if you need this dataset for study purpose, or use COCO keypoint dataset instead.)

### Police Gestures
Download PoliceGesture dataset from

[Google Drive](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym?usp=sharing)

or

[Nutstore 坚果云](https://www.jianguoyun.com/p/DQFgxv8Q9_LMBhiVrvYB)


Put corresponding dataset into:

(`home` is 'C:\Users\(name)' in Windows and `/home/(name)` in Ubuntu)

```
(home folder)/PoliceGestureLong
(home folder)/AI_challenger_keypoint
```

## Install requirements:

```bash
# Python 3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install visdom opencv-python imgaug ujson
```

## Usage
```bash
usage: ctpgr.py [-h] [--train_keypoint] [--clean_saved_skeleton] [--train_gcn_fc] [--train_gcn_lstm] [--train_bla_lstm] [--eval_gcn_fc] [--eval_gcn_lstm]
                [--eval_bla_lstm]

optional arguments:
  -h, --help            show this help message and exit
  --train_keypoint      Train human keypoint estimation model from ai_challenger dataset
  --clean_saved_skeleton
                        Clean the saved skeleton from generated/coords to regenerate them during next training
  --train_gcn_fc        Train GCN_FC (proposed method)
  --train_gcn_lstm      Train GCN_LSTM (for ablation experiment)
  --train_bla_lstm      Train BLA_LSTM (baseline, for ablation)
  --eval_gcn_fc         Compute Jaccard score of GCN_FC (proposed method)
  --eval_gcn_lstm       Compute Jaccard score of GCN_LSTM (for ablation experiment)
  --eval_bla_lstm       Compute Jaccard score of BLA_LSTM (baseline, for ablation)

```

The Jaccard score plotting codes **only exist in [release version](https://github.com/zc402/ctpgr-pytorch/releases/)** at:
```
docs/result_compare.ipynb

docs/result_compare_class.ipynb
```
The confusion matrix is located at:
```
docs/cm.pdf
```