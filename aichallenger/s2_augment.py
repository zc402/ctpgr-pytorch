import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage

from pathlib import Path
import cv2
import numpy as np

from aichallenger import AicResize
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List

# TODO: random seeds for threads
class AicAugment(AicResize):
    """
    AicAugment dataset provides augmented images and labels.
    Construct 'aug_img' and 'aug_label'
    Add 'img_before_aug' and 'img_after_aug' if visual_debug is True
    """

    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, **kwargs)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        if self.is_train:
            img_aug, pts_aug, debug_dict = self.__image_augment(res_dict['resized_img'], res_dict['resized_label'])
            res_dict.update(debug_dict)
        else:
            img_aug, pts_aug = res_dict['resized_img'], res_dict['resized_label']
        res_dict['aug_img'] = img_aug
        res_dict['aug_label'] = pts_aug
        return res_dict

    def __image_augment(self, image: np.ndarray, crowd: Crowd):
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
        aug_img, aug_pts, aug_boxes, debug_dict = self.__aug_with_points(image, pts_before_aug, bbs_before_aug)

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

        return aug_img, aug_labels, debug_dict

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
                rotate=(-10, 10),
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            )
        ])
        det = seq.to_deterministic()
        image_aug = det.augment_image(image)
        kps_aug = det.augment_keypoints(kps)
        bbs_aug = det.augment_bounding_boxes(bbs)
        # Show augmented points on image
        debug_dict = {}
        if 'visual_debug' in self.kwargs and self.kwargs.get('visual_debug'):
            image_before = kps.draw_on_image(image, size=5)
            image_before = bbs.draw_on_image(image_before, size=2)
            image_after = kps_aug.draw_on_image(image_aug, size=5)
            image_after = bbs_aug.draw_on_image(image_after, size=2)
            debug_dict['img_before_aug'] = image_before
            debug_dict['img_after_aug'] = image_after
        return image_aug, kps_aug, bbs_aug, debug_dict


