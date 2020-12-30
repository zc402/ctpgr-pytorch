from aichallenger import AicNative
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage

class AicResize(AicNative):
    """
    Provides resized images for network input
    Construct 'resized_img' and 'resized_label'
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, **kwargs):
        super().__init__(data_path, is_train, **kwargs)
        self.resize_img_size = resize_img_size
        # self.__fix_ratio_resize = FixRatioImgResize(resize_img_size)
        self.rkr = ResizeKeepRatio(resize_img_size)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        # 图像Resize至网络输入大小
        img = res_dict['native_img']
        crowd = res_dict['native_label']

        boxes = np.asarray([p.box for p in crowd], np.int)
        joints = np.asarray([[[j.x, j.y] for j in p.joints] for p in crowd], np.int)

        img_aug, np_kps_aug, np_boxes_aug = self.rkr.resize(img, joints, boxes)

        num_people: int = len(crowd)
        num_joints: int = len(crowd[0].joints)
        aug_labels: Crowd = []
        for p in range(num_people):
            np_box = np_boxes_aug[p]
            box = Box(*np_box)
            joints = []
            for j in range(num_joints):
                v = crowd[p].joints[j].v
                x, y = np_kps_aug[p][j]
                joint = Joint(x, y, v)
                joints.append(joint)
            person = Person(box, joints)
            aug_labels.append(person)
        res_dict['resized_img'] = img_aug
        res_dict['resized_label'] = aug_labels

        return res_dict

class ResizeKeepRatio:
    def __init__(self, target_size: tuple):
        self.target_size = target_size
        assert len(self.target_size) == 2, 'Expect tuple (w, h) for target size'

    def resize(self, img, pts: np.ndarray, boxes: np.ndarray):
        pts_shape = pts.shape
        pts = pts.reshape((-1, 2))
        boxes_shape = boxes.shape
        boxes = boxes.reshape((-1, 4))
        # old_kps = [Keypoint(x,y) for x,y in pts]
        # old_boxes = [BoundingBox(x1, y1, x2, y2) for x1, y1, x2, y2 in boxes]

        tw, th = self.target_size
        ih, iw, ic = img.shape
        kps_on_image = KeypointsOnImage.from_xy_array(pts, shape=img.shape)
        boxes_on_img = BoundingBoxesOnImage.from_xyxy_array(boxes, shape=img.shape)

        seq = self.__aug_sequence((iw, ih), (tw, th))
        det = seq.to_deterministic()
        img_aug = det.augment_image(img)
        kps_aug = det.augment_keypoints(kps_on_image)
        boxes_aug = det.augment_bounding_boxes(boxes_on_img)

        np_kps_aug = kps_aug.to_xy_array()
        # np_kps_aug = np.array([(p.x, p.y) for p in kps_aug])
        np_kps_aug = np_kps_aug.reshape(pts_shape)
        np_boxes_aug = boxes_aug.to_xy_array()
        # np_boxes_aug = np.array([(p.x1, p.y1, p.x2, p.y2) for p in boxes_aug])
        np_boxes_aug = np_boxes_aug.reshape(boxes_shape)
        return img_aug, np_kps_aug, np_boxes_aug


    def __aug_sequence(self, input_size, target_size):
        iw, ih = input_size
        i_ratio = iw / ih
        tw, th = target_size
        t_ratio = tw / th
        if i_ratio > t_ratio:
            # Input image wider than target, resize width to target width
            resize_aug = iaa.Resize({"width": tw, "height": "keep-aspect-ratio"})
        else:
            # Input image higher than target, resize height to target height
            resize_aug = iaa.Resize({"width": "keep-aspect-ratio", "height": th})
        pad_aug = iaa.PadToFixedSize(width=tw, height=th, position="center")
        seq = iaa.Sequential([resize_aug, pad_aug])
        return seq



