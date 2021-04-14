# 中国交通警察指挥手势识别
This is a pytorch deep learning project that recognizes 8 kinds of Traffic police commanding gestures.

**[English Readme](readme.en.md)**

识别8种中国交通警察指挥手势的Pytorch深度学习项目

<p align="center">
    <img src="docs/intro.gif" width="480">
</p>

## 论文
电子学报： http://www.ejournal.org.cn/CN/10.3969/j.issn.0372-2112.2020.05.018 

## 安装

### 下载模型参数文件`checkpoint`和生成的骨架`generated`
下载地址：

[GoogleDrive](https://drive.google.com/drive/folders/1kngUBiiUWUOt1NeasHS9IMGQvJrFoxpO?usp=sharing)

[Nutstore 坚果云](https://www.jianguoyun.com/p/DQz4eNMQ9_LMBhi-9dYD)

放置在:

```
ctpgr-pytorch/checkpoints
ctpgr-pytorch/generated
```


### 下载交警手势数据集（必选）和AI Challenger数据集（可选）

交警手势数据集下载：

[Google Drive](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym?usp=sharing)

[Nutstore 坚果云](https://www.jianguoyun.com/p/DQFgxv8Q9_LMBhiVrvYB)

放置在：
```
(用户文件夹)/PoliceGestureLong
(用户文件夹)/AI_challenger_keypoint

# 用户文件夹 在 Windows下是'C:\Users\(用户名)'，在Linux下是 '/home/(用户名)'
```

### 安装Pytorch和其它依赖：
```bash
# Python 3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install visdom opencv-python imgaug ujson sklearn
```

## 使用
```bash
# 识别自定义视频文件 
python ctpgr.py -p C:\012.mp4

# 识别摄像头实时视频
python ctpgr.py -r

# 识别交警数据集中test文件夹第0个视频
python ctpgr.py -b 0

# 训练等其它功能见帮助
python ctpgr.py --help
```
