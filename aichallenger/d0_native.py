from typing import List
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
import cv2

from aichallenger.defination import Box, Joint, Person, Crowd


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

        crowd: Crowd = []
        for k, v in human_annotations.items():
            box = Box(*v)
            joint_list = keypoint_annotations[k]  # x1 y1 v1 x2 y2 v2
            joint_list = np.array(joint_list).reshape(14, 3)
            joints: List[Joint] = []
            for joint_xyv in joint_list:
                joint = Joint(*joint_xyv)
                joints.append(joint)
            person = Person(box, joints)
            crowd.append(person)

        return {'native_img': native_image, 'native_label': crowd}





