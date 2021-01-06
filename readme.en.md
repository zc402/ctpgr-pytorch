# The Chinese Traffic Police Gesture Recognizer
This is a pytorch deep learning project that recognizes 8 kinds of Traffic police commanding gestures.

**[中文 Readme](readme.md)**

## Install 

### Download parameters `checkpoint` and `generated`

[GoogleDrive](https://drive.google.com/drive/folders/1kngUBiiUWUOt1NeasHS9IMGQvJrFoxpO?usp=sharing)

[Nutstore 坚果云](https://www.jianguoyun.com/p/DQz4eNMQ9_LMBhi-9dYD)

Put them into:

```
ctpgr-pytorch/checkpoints
ctpgr-pytorch/generated
```

### Download PoliceGesture dataset (required) and AI Challenger dataset (optional), put them into

[Google Drive](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym?usp=sharing)

[Nutstore 坚果云](https://www.jianguoyun.com/p/DQFgxv8Q9_LMBhiVrvYB)


```
(home folder)/PoliceGestureLong
(home folder)/AI_challenger_keypoint
# home folder is 'C:\Users\(name)' in Windows and `/home/(name)` in Ubuntu
```

### Install pytorch and other requirements:

```bash
# Python 3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install ujson
pip install visdom opencv-python imgaug
```

## Usage
```bash
# Recognize gestures in custom video
python ctpgr.py -p C:\012.mp4

# Recognize gestures in realtime
python ctpgr.py -r

# Recognize gestures in 1st video of test set
python ctpgr.py -b 0

# Read help to see other features
python ctpgr.py --help
```