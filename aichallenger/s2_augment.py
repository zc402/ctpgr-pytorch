import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage

from pathlib import Path
import cv2
import numpy as np

from aichallenger import AicResize
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List

from constants.enum_keys import HK


class AicAugment(AicResize):

    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, **kwargs)
        self.aug = ImgAugmentor()

    def __getitem__(self, index) -> dict:
        res = super().__getitem__(index)
        # 图像Resize至网络输入大小
        img = res[HK.RE_IMAGE]
        if self.is_train:
            img_aug, np_kps_aug, np_boxes_aug = self.aug.aug(img, res[HK.RE_KEYPOINTS], res[HK.RE_BOXES])
        else:
            img_aug, np_kps_aug, np_boxes_aug = img, res[HK.RE_KEYPOINTS], res[HK.RE_BOXES]

        res[HK.AUG_IMAGE] = img_aug
        res[HK.AUG_KEYPOINTS] = np_kps_aug
        res[HK.AUG_BOXES] = np_boxes_aug

        if 'visual_debug' in self.kwargs and self.kwargs.get('visual_debug'):

            img_draw = KeypointsOnImage.from_xy_array(res[HK.AUG_KEYPOINTS].reshape(-1, 2), shape=img_aug.shape)\
                .draw_on_image(img_aug, size=5)
            img_draw = BoundingBoxesOnImage.from_xyxy_array(res[HK.AUG_BOXES].reshape(-1, 4), shape=img_aug.shape)\
                .draw_on_image(img_draw, size=2)
            res[HK.DEBUG_AUG_IMAGE] = img_draw

        return res


class ImgAugmentor:

    def aug(self, img, pts: np.ndarray, boxes: np.ndarray):
        pts_shape = pts.shape
        pts = pts.reshape((-1, 2))
        boxes_shape = boxes.shape
        boxes = boxes.reshape((-1, 4))

        kps_on_image = KeypointsOnImage.from_xy_array(pts, shape=img.shape)
        boxes_on_img = BoundingBoxesOnImage.from_xyxy_array(boxes, shape=img.shape)

        seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.2)),  # change brightness
            iaa.Affine(
                rotate=(-5, 5),
                scale=(0.9, 1.05),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            ),
            iaa.GaussianBlur(sigma=(0, 0.7)),
            iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 7)))
        ])
        det = seq.to_deterministic()
        img_aug = det.augment_image(img)
        kps_aug = det.augment_keypoints(kps_on_image)
        boxes_aug = det.augment_bounding_boxes(boxes_on_img)

        np_kps_aug = kps_aug.to_xy_array()
        np_kps_aug = np_kps_aug.reshape(pts_shape)
        np_boxes_aug = boxes_aug.to_xy_array()
        np_boxes_aug = np_boxes_aug.reshape(boxes_shape)

        return img_aug, np_kps_aug, np_boxes_aug



