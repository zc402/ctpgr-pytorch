from typing import List
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import ujson as json
import numpy as np
import cv2

from aichallenger.defines import Box, Joint, Person, Crowd
from constants.enum_keys import HK


class AicNative(Dataset):
    """
    Basic AI Challenger dataset loads images and labels
    Construct 'native_img' and 'native_label'
    """

    def __init__(self, data_path: Path, is_train: bool, **kwargs):
        self.is_train = is_train
        self.kwargs = kwargs

        paths = dict()
        paths[("train", "root")] = data_path / "ai_challenger_keypoint_train_20170909"
        paths[("train", "json")] = paths[("train", "root")] / "keypoint_train_annotations_20170909.json"
        paths[("train", "images")] = paths[("train", "root")] / "keypoint_train_images_20170902"

        paths[("val", "root")] = data_path / "ai_challenger_keypoint_validation_20170911"
        paths[("val", "json")] = paths[("val", "root")] / "keypoint_validation_annotations_20170911.json"
        paths[("val", "images")] = paths[("val", "root")] / "keypoint_validation_images_20170911"

        if is_train:
            with open(paths[("train", "json")]) as json_file:
                labels = json.load(json_file)
            paths[("current", "images")] = paths[("train", "images")]
        else:
            with open(paths[("val", "json")]) as json_file:
                labels = json.load(json_file)
            paths[("current", "images")] = paths[("val", "images")]

        self.__paths = paths
        self.__labels = labels

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, index) -> dict:
        image_name = self.__labels[index]["image_id"] + ".jpg"
        image_path = self.__paths["current", "images"] / image_name
        native_image = cv2.imread(str(image_path))

        keypoint_annotations = self.__labels[index]["keypoint_annotations"]
        human_annotations = self.__labels[index]["human_annotations"]

        num_people = len(human_annotations.keys())
        num_joints = 14
        keypoints = []
        visibilities = []
        boxes = []
        for person_key, box_x4 in human_annotations.items():
            boxes.append(box_x4)
            point_xyv = keypoint_annotations[person_key]  # x1 y1 v1 x2 y2 v2
            point_x = point_xyv[0::3]
            point_y = point_xyv[1::3]
            point_v = point_xyv[2::3]
            keypoints.append(list(zip(point_x, point_y)))
            visibilities.append(point_v)

        keypoints = np.array(keypoints, dtype=np.float32).reshape((num_people, num_joints, 2))
        visibilities = np.array(visibilities, dtype=np.int).reshape((num_people, num_joints))
        boxes = np.array(boxes, dtype=np.float32).reshape((num_people, 4))

        return {HK.NATIVE_IMAGE: native_image, HK.BOXES: boxes, HK.KEYPOINTS: keypoints,
                HK.VISIBILITIES: visibilities, HK.NUM_PEOPLE: num_people}







