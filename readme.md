## The Chinese Traffic Police Gesture Recognizer

### 中国交通警察指挥手势识别

This is a pytorch deep learning project that recognizes 8 kinds of Traffic police commanding gestures.

## Quick Start
#### Download checkpoints from GoogleDrive[], put them into: 

从GoogleDrive下载模型参数，放置在：
```
(project folder)/checkpoints/pose_model.pt
(project folder)/checkpoints/lstm.pt
```

#### Download PoliceGesture dataset and AI Challenger dataset (optional), put them into

下载交警手势数据集和AI Challenger数据集（可选），放置在：
```
(home folder)/PoliceGestureLong
(home folder)/AI_challenger_keypoint
# home folder is 'C:\Users\(name)' in Windows and `/home/(name)` in Ubuntu
```

#### Install pytorch and other requirements:

安装Pytorch和其它依赖：
```
# Python 3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install ujson
pip install visdom opencv-python imgaug
```

#### Run
Recognize video 0 in test folder

识别交警数据集中test文件夹第0个视频
```
python ctpgr -b 0
```

Recognize a custom video

识别自定义视频文件
```
python ctpgr -p C:\Users\zc\PoliceGestureLong\test\012.mp4
```