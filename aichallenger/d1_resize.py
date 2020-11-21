from aichallenger.d0_native import AicNative
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List


class AicResize(AicNative):
    """
    Provides resized images for network input
    Construct 'resized_img' and 'resized_label'
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, **kwargs):
        super().__init__(data_path, is_train, **kwargs)
        self.resize_img_size = resize_img_size
        self.__fix_ratio_resize = FixRatioImgResize(resize_img_size)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        # 图像Resize至网络输入大小
        resized_image, ratio = self.__fix_ratio_resize.resize(res_dict['native_img'])

        # 将人物关键点Resize
        resized_crowd: List[Person] = []
        for person in res_dict['native_label']:
            new_person = self.__resize_person_labels(person, ratio)
            resized_crowd.append(new_person)

        res_dict['resized_img'] = resized_image
        res_dict['resized_label'] = resized_crowd

        return res_dict

    @staticmethod
    def __resize_person_labels(person: Person, ratio: float) -> Person:
        new_joints = [Joint(round(j.x * ratio), round(j.y * ratio), j.v) for j in person.joints]
        new_box = Box(*[round(c * ratio) for c in person.box])
        new_person = Person(box=new_box, joints=new_joints)
        return new_person


class FixRatioImgResize:
    """
    Load and resize image with original ratio fixed
    """

    def __init__(self, new_size: tuple):
        self.new_size = new_size
        assert len(self.new_size) == 2, "incorrect new_size shape, should be (w,h)"

    def resize(self, native_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image with fixed ratio and black padding
        :param native_img: PIL Image
        :param new_size: (new_w, new_h)
        :return: (new_image, ratio)
        ratio is new/native
        """
        new_w, new_h = self.new_size  # 预计输入网络的图片尺寸
        w_div_h = new_w / new_h  # w divide by h
        native_h, native_w = native_img.shape[:2]  # 原始图片尺寸
        native_w_div_h = native_w / native_h
        if native_w_div_h > w_div_h:
            # 原图比例宽边为主，需缩短/拉伸宽边到新图宽边
            ratio = new_w / native_w  # 新图/原图 的比例
            resized_w = new_w  # 原始图片缩放后Padding之前的尺寸
            resized_h = native_h * ratio
        else:
            # 原图比例高边为主，缩短/拉伸高边到新图高边
            ratio = new_h / native_h
            resized_w = native_w * ratio
            resized_h = new_h

        resized_image = cv2.resize(native_img, tuple((int(resized_w), int(resized_h))), interpolation=cv2.INTER_LINEAR)
        channels = native_img.shape[2]
        canvas = np.zeros((new_w, new_h, channels), dtype=np.uint8)
        # Paste
        canvas[:resized_image.shape[0], :resized_image.shape[1]] = resized_image
        return canvas, ratio


