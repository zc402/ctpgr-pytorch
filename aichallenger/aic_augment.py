import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage

from aichallenger.aic_resize import AicResize
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defination import Box, Joint, Person, Crowd
from typing import Tuple, List


class AicAugment(AicResize):
    """
    AicAugment dataset provides augmented images and labels.
    """

    def __init__(self, data_path: Path, is_train: bool, img_resize: tuple):
        super().__init__(data_path, is_train, img_resize)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        if self.is_train:
            img_aug, pts_aug = self.image_augment(res_dict['resized_img'], res_dict['resized_label'])
        else:
            img_aug, pts_aug = res_dict['resized_img'], res_dict['resized_label']
        res_dict['aug_img'] = img_aug
        res_dict['aug_label'] = pts_aug
        return res_dict

    def image_augment(self, image: np.ndarray, crowd: Crowd):
        """
        Augment an image with keypoints.
        Flatten the box and joints for imgaug, then collect them back
        """
        num_people: int = len(crowd)
        num_joints: int = len(crowd[0].joints)

        boxes = [p.box for p in crowd]
        joints = [[[(j.x, j.y)] for j in p.joints] for p in crowd]

        boxes_flat = np.array(boxes, np.int).reshape(-1, 4)
        joints_flat = np.array(joints, np.int).reshape(-1, 2)

        pts_before_aug = [Keypoint(*c) for c in joints_flat]
        bbs_before_aug = [BoundingBox(*c) for c in boxes_flat]
        # Image augmentation with keypoints
        aug_img, aug_pts, aug_boxes = self.__aug_with_points(image, pts_before_aug, bbs_before_aug)

        np_aug_pts_flat = np.array([(p.x, p.y) for p in aug_pts])
        np_aug_boxes = np.array([(p.x1, p.x2, p.y1, p.y2) for p in aug_boxes])
        # Recover flattened points

        aug_joints = np_aug_pts_flat.reshape((num_people, num_joints, 2))

        aug_labels: Crowd = []
        for p in range(num_people):
            np_box = np_aug_boxes[p]
            box = Box(*np_box)
            joints = []
            for j in range(num_joints):
                v = crowd[p].joints[j].v
                x, y = aug_joints[p][j]
                joint = Joint(x, y, v)
                joints.append(joint)
            person = Person(box, joints)
            aug_labels.append(person)

        return aug_img, aug_labels

    def __aug_with_points(self, image: np.ndarray, keypoint_list: List[Keypoint], box_list: List[BoundingBox]):
        """
        Image and keypoints augmentation
        Arg:
            image (cv2 image)
            keypoint_list (list of imgaug.augmentables.Keypoint)
        """

        kps = KeypointsOnImage(keypoint_list, shape=image.shape)
        bbs = BoundingBoxesOnImage(box_list, shape=image.shape)
        seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.5)),  # change brightness
            iaa.Affine(
                rotate=(-30, 30),
                scale=(0.7, 1.3),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            )
        ])
        det = seq.to_deterministic()
        image_aug = det.augment_image(image)
        kps_aug = det.augment_keypoints(kps)
        bbs_aug = det.augment_bounding_boxes(bbs)
        # Apply augmentation
        self.visual_debug = True
        if self.visual_debug:
            image_before = kps.draw_on_image(image, size=7)
            image_after = kps_aug.draw_on_image(image_aug, size=7)
            cv2.imshow("image_before", image_before)
            cv2.imshow("image_after", image_after)
            cv2.waitKey()
        return image_aug, kps_aug, bbs_aug


