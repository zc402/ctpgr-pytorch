from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from collections import OrderedDict
import torch.utils.data as data
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
from pathlib import Path
from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2

class AICDataset(data.Dataset):
    """Dataloader for AI Challenger Poses"""

    def __init__(self, data_path, input_img_size, heat_size, is_train):
        """

        :param data_path:
        :param input_img_size:
        :param heat_size:
        :param is_train:
        """
        self.data_path = data_path  # Root path of dataset
        self.input_img_size = input_img_size  # The size of the image being input to the neural network
        self.heat_size = heat_size
        # is_train: Set data path to "train" or "val" folder; disable augmentation when val
        self.is_train = is_train
        self.theta = 4  # Theta for gaussian kernel
        self.visual_debug = False  # Show image and joint heatmaps (vis and vis_or_not)
        # Default input normalization for model parameters in torchvision model zoo
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.fix_ratio_resize = FixRatioImgResize(input_img_size)
        self.aic_augmentor = AICAugment()
        self.heat_generator = HeatmapGenerator(heat_size, self.theta)

        assert input_img_size[0] % heat_size[0] == 0 and input_img_size[1] % heat_size[1] == 0, \
            "Incorrect sizes: input_img_size must be divisible by heat_size"


        # Load labels
        paths = OrderedDict()
        paths[("train", "root")] = self.data_path / "ai_challenger_keypoint_train_20170909"
        paths[("train", "json")] = paths[("train", "root")] / "keypoint_train_annotations_20170909.json"
        paths[("train", "images")] = paths[("train", "root")] / "keypoint_train_images_20170902"

        paths[("val", "root")] = self.data_path / "ai_challenger_keypoint_validation_20170911"
        paths[("val", "json")] = paths[("val", "root")] / "keypoint_validation_annotations_20170911.json"
        paths[("val", "images")] = paths[("val", "root")] / "keypoint_validation_images_20170911"

        if is_train:
            with open(paths[("train", "json")]) as json_file:
                labels = json.load(json_file)
                paths[("current", "images")] = paths[("train", "images")]
        else:
            with open(paths["val", "json"]) as json_file:
                labels = json.load(json_file)
                paths[("current", "images")] = paths[("val", "images")]

        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """Randomly return one image and corresponding label
        -> Input
        -> Resize to network input size
        -> Augmentation
        -> Resize label to heat size
        -> Generate Heatmap
        -> Output
        """
        # "keypoint_annotations": {"human3": [0, 0, 3, 0, 0, 3, 0, 0, 3, 67, 279, 1 ... }

        keypoint_annotations = self.labels[index]["keypoint_annotations"]
        image_name = self.labels[index]["image_id"] + ".jpg"
        image_path = self.paths["current", "images"] / image_name
        native_image = self.pil_loader(image_path)
        # 图像Resize至网络输入大小
        resized_image, ratio = self.fix_ratio_resize.resize(native_image)

        # 将每人物关键点Resize
        # 输出person_pts shape: (person, 14, 3)
        person_pts = [np.array(v).reshape(14, 3) for k, v in keypoint_annotations.items()]
        person_pts = np.array(person_pts, dtype=np.int)
        for joint_pts in person_pts:
            for pt in joint_pts:
                x, y = pt[0:2]
                # 将Keypoint按照给定ratio进行缩放
                new_x, new_y = x * ratio, y * ratio
                pt[0] = new_x
                pt[1] = new_y

        if self.is_train:
            img_aug, pts_aug = self.image_augment(resized_image, person_pts)
        else:
            img_aug, pts_aug = resized_image, person_pts

        heatmap = self.gen_AIC_confidence_map(pts_aug)  # Dict: confidence, visible

        if self.visual_debug:
            self.show_heatmap(img_aug, heatmap)

        # scale to [0,1) and transpose HWC to CHW
        resized_image = transforms.ToTensor()(resized_image)
        # To use torchvision pretrained model, the input image must be normalized as follow.
        resized_image = self.normalize(resized_image)

        for key, val in heatmap.items():  # Heatmap to tensor
            heatmap[key] = torch.from_numpy(np.asarray(val))

        output = {"resized": resized_image, "heatmap": heatmap}
        return output

    def image_augment(self, image, person_pts):
        """
        对图像和关键点进行Augment
        Args:
            image (PIL Image): PIL Image
            person_pts (numpy): keypoints shape=(person, joints, xyv)

        Returns:
            img_aug (PIL Image)
            person_pts_aug (numpy): shape=(person, joints, xyv)
        """
        # 转换关键点至Imgaug关键点格式，转换PIL Image 至 OpenCV格式，给Imgaug使用
        np_pts_flat = person_pts[:, :, 0:2].reshape((-1, 2))
        ia_pts = [Keypoint(x, y) for x, y in np_pts_flat]
        cv_resized_img = np.array(image)
        # 转换至cv图片格式
        cv_resized_img = cv_resized_img[:, :, ::-1].copy()  # Convert RGB to BGR
        image_aug, kps_aug = self.aic_augmentor.aug(cv_resized_img, ia_pts)
        # 还原关键点格式
        kps_aug = [(a.x, a.y) for a in kps_aug]
        num_people = person_pts.shape[0]
        kps_aug = np.array(kps_aug).reshape((num_people, 14, 2))
        visibility = person_pts[:, :, 2:3]
        person_pts_aug = np.concatenate((kps_aug, visibility), axis=2)
        assert person_pts_aug.shape == (num_people, 14, 3)
        # 还原图像格式至PIL
        image_aug = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
        image_aug = Image.fromarray(image_aug)
        return image_aug, person_pts_aug

    def gen_AIC_confidence_map(self, person_pts):
        """
        为AIC数据集生成 confidence heatmap
        :param person_pts: numpy array shape=(person, joint, xyv)
        :param input_size: Image size for convolutional network input
        :param heat_size: Heatmap size for network loss
        :return:
        """
        assert len(person_pts.shape) == 3

        num_people = person_pts.shape[0]
        # The "v" in (x,y,v) Annotation means:
        # Visible: 1, Not visible but labeled: 2, Not visible and not labeled: 0
        zero_heat = np.zeros((self.heat_size[1], self.heat_size[0]), np.float)  # When no keypoint appears
        heat = {"vis_or_not": [], "visible": []}  # shape: (J,H,W). on_image: keypoint on image, visible or not visible
        for j in range(14):  # Joint

            # Heatmaps for same joint and different person
            heatmaps_vis_or_not = []  # Has heat when v=1,2, No heat when v=0 (not labeled)
            heatmaps_visible = []  # Has heat when v = 1, No heat when v=0,2 (not visible, labeled/unlabeled)
            for p in range(num_people):  # People
                cx, cy, v_value = person_pts[p, j]
                # cx, cy = cx/w_scale, cy/h_scale
                is_labeled = (v_value == 1 or v_value == 2)
                is_visible = (v_value == 1)

                heatmap = self.heat_generator.gen_heat_adjust_pt(self.input_img_size, (cx, cy))

                heat_conf = heatmap if is_labeled else zero_heat
                heat_vis = heatmap if is_visible else zero_heat

                heatmaps_vis_or_not.append(heat_conf)
                heatmaps_visible.append(heat_vis)
            heat_people_conf = np.amax(heatmaps_vis_or_not, axis=0) # shape:(H,W). Heatmap showing 1 joint of multiple people
            heat_people_vis = np.amax(heatmaps_visible, axis=0)
            heat["vis_or_not"].append(heat_people_conf)
            heat["visible"].append(heat_people_vis)

        return heat

    def show_heatmap(self, img, heat):
        conf_gray_map = np.amax(heat["vis_or_not"], axis=0)
        vis_gray_map = np.amax(heat["visible"], axis=0)

        plt.close('all')
        fig, ax = plt.subplots(2, 2)
        img = np.asarray(img)
        ax[0][0].imshow(img)
        ax[1][0].imshow(conf_gray_map)
        ax[1][1].imshow(vis_gray_map)
        plt.show()
        plt.pause(10)

    def pil_loader(self, image_path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class FixRatioImgResize:
    """
    Load and resize image to designated size, but with original ratio fixed
    """

    def __init__(self, new_size):
        self.new_size = new_size
        assert len(self.new_size) == 2, "incorrect new_size shape, should be (w,h)"

    def resize(self, native_img):
        """
        Resize image with fixed ratio and black padding
        :param native_img: PIL Image
        :param new_size: (new_w, new_h)
        :return: (new_image, ratio)
        ratio is actual resized ratio of new:native
        """
        new_w, new_h = self.new_size  # 预计输入网络的图片尺寸
        w_div_h = new_w / new_h  # w divide by h
        native_w, native_h = native_img.size  # 原始图片尺寸
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

        resized_image = native_img.resize((int(resized_w), int(resized_h)), Image.BILINEAR)
        background = Image.new("RGB", (new_w, new_h))
        background.paste(resized_image)
        return background, ratio


class AICAugment:

    def __init__(self):
        # visual_debug: Show original and augmented images with keypoints painted
        self.visual_debug = False

    def aug(self, image, keypoint_list):
        """
        Image and keypoints augmentation
        Arg:
            image (cv2 image)
            keypoint_list (list of imgaug.augmentables.Keypoint)
        """

        kps = KeypointsOnImage(keypoint_list, shape=image.shape)
        seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.5)),  # change brightness
            iaa.Affine(
                rotate=(-30, 30),
                scale=(0.7, 1.3),
                translate_percent = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            )
        ])
        # Apply augmentation
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        if self.visual_debug:
            image_before = kps.draw_on_image(image, size=7)
            image_after = kps_aug.draw_on_image(image_aug, size=7)
            cv2.imshow("image_before", image_before)
            cv2.imshow("image_after", image_after)
            cv2.waitKey()
        return image_aug, kps_aug


class HeatmapGenerator:

    def __init__(self, heat_size, theta):
        self.w = heat_size[0]  # Heat w
        self.h = heat_size[1]
        self.theta = theta

    def gen_heat(self, pt):
        """
        Args:
            pt: a coordinate (x,y)
        """
        cx, cy = pt[0], pt[1]
        x_mesh, y_mesh = np.meshgrid(np.arange(0, self.w), np.arange(0, self.h))
        heatmap = np.exp(-(np.square(x_mesh - cx) + np.square(y_mesh - cy)) / (np.square(self.theta)))
        return heatmap

    def gen_heat_adjust_pt(self, img_size, pt):
        """
        When image size not equal to heat size, this function adjust image keypoints to heat keypoints
        :param img_size:
        :return:
        """
        img_w, img_h = img_size
        ratio_w = self.w / img_w
        ratio_h = self.h / img_h
        heat_x = pt[0] * ratio_w
        heat_y = pt[1] * ratio_h
        heatmap = self.gen_heat((heat_x, heat_y))
        return heatmap