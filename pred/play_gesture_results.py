from pathlib import Path

import cv2
import numpy as np
from imgaug import KeypointsOnImage
from imgaug.imgaug import draw_text

from constants.enum_keys import PG
from pgdataset.s1_skeleton_coords import SkeletonCoordsDataset
import pred.gesture_pred

class Player:
    def __init__(self):
        self.img_size = (512, 512)
        self.gpred = pred.gesture_pred.GesturePred()

    def play(self, is_train, video_index):
        self.scd = SkeletonCoordsDataset(Path.home() / 'PoliceGestureLong', is_train, self.img_size)
        res = self.scd[video_index]
        coord_norm_FXJ = res[PG.COORD_NORM]  # Shape: F,X,J
        coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
        coord = coord_norm_FJX * np.array(self.img_size)
        img_shape = self.img_size[::-1] + (3,)
        kps = [KeypointsOnImage.from_xy_array(coord_JX, shape=img_shape) for coord_JX in coord]  # (frames, KeyOnImage)
        cap = cv2.VideoCapture(str(res[PG.VIDEO_PATH]))
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(1000/v_fps)
        for n in range(v_size):
            ret, img = cap.read()
            re_img = cv2.resize(img, self.img_size)
            gdict = self.gpred.from_skeleton(coord_norm_FXJ[n][np.newaxis])
            gesture = gdict[PG.OUT_ARGMAX]
            re_img = draw_text(re_img, 50, 100, str(gesture))
            pOnImg = kps[n]
            img_kps = pOnImg.draw_on_image(re_img)
            cv2.imshow("Play saved keypoint results", img_kps)
            cv2.waitKey(duration)
