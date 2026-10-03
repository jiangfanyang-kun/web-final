# author  : jerrylee
# date    : 2022/2/15
# time    : 18:42
# encoding: utf-8
import json
import os

import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils_yolo.datasets import letterbox
from utils_yolo.general import non_max_suppression, scale_coords
from utils_yolo.torch_utils import select_device


class Detector:

    def __init__(self):
        """
        初始化相关配置及其权重
        """
        base_path = os.path.dirname(os.getcwd())
        config_path = base_path + r'\web_final-master\src\templates\detector_config.json'
        with open(config_path, 'rb') as _config:
            config = json.load(_config)
        self.img_size = config['img_size']  # 大小
        self.threshold = config['threshold']  # 置信度
        self.stride = config['stride']  # 步长
        self.weights = "yolo_weight/" + config['weight']  # 权重文件
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(
            self.weights,
            map_location=self.device
        )

        model.to(
            self.device
        ).eval()

        # ==========================================================
        # RTX显卡使用FP16推理
        # CPU继续使用FP32
        # ==========================================================

        self.half = (
                self.device.type != 'cpu'
        )

        if self.half:

            model.half()

            print(
                '[YOLO] 使用 FP16 CUDA 推理'
            )

        else:

            model.float()

            print(
                '[YOLO] 使用 FP32 CPU 推理'
            )

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB
        img = np.ascontiguousarray(img)  # 地址连续化
        img = torch.from_numpy(img).to(self.device)
        if self.half:
            img = img.half()
        else:
            img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        # 推理阶段不计算梯度
        with torch.no_grad():

            pred = self.m(
                img,
                augment=False
            )[0]

        # NMS继续使用float32，兼容旧版YOLOv5
        pred = pred.float()

        pred = non_max_suppression(
            pred,
            self.threshold,
            0.4
        )
        boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ['person', 'car', 'bus', 'truck']:
                        # if lbl not in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes


def shot_accident(img, bboxes):
    base_path = os.path.dirname(os.getcwd())
    save_path = base_path + r'\web_final-master\src\static\save_accident'
    if len(bboxes) > 0:
        for x1, y1, x2, y2, _, _id, _ in bboxes:
            c1, c2 = (x1, y1), (x2, y2)
            crop = img[c1[1]:c2[1] - c2[0], c2[0]:c2[1] - c1[0]]
            cv2.imwrite(save_path + r'\{}.jpg'.format(_id), crop)


def shot_img(img, c1, c2, track_id, label='', scale_x=1.0, scale_y=1.0):
    """保存车辆/行人的高清撞线截图"""

    if img is None or img.size == 0:
        return False

    img_h, img_w = img.shape[:2]

    # 检测坐标映射回原始高清画面
    x1 = int(c1[0] * scale_x)
    y1 = int(c1[1] * scale_y)
    x2 = int(c2[0] * scale_x)
    y2 = int(c2[1] * scale_y)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # 按目标尺寸自动增加约 25% 留白
    pad_x = max(4, int(box_w * 0.12))
    pad_y = max(4, int(box_h * 0.12))

    # 边界保护，避免负数坐标导致异常裁图
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return False

    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return False

    # 小目标适当放大，方便网页查看
    crop_h, crop_w = crop.shape[:2]
    if crop_w < 240:
        ratio = 240.0 / crop_w
        new_h = max(1, int(crop_h * ratio))
        crop = cv2.resize(
            crop,
            (240, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    # 在截图上显示类别和追踪 ID
    title = '{} ID-{}'.format(label, track_id)

    cv2.rectangle(
        crop,
        (0, 0),
        (min(crop.shape[1], 180), 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        crop,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # 不再依赖 web_final / web_final-master 文件夹名
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    save_path = os.path.join(
        project_root,
        'src',
        'static',
        'save_img'
    )

    os.makedirs(save_path, exist_ok=True)

    path = os.path.join(
        save_path,
        '{}.jpg'.format(track_id)
    )
    # 同一个追踪 ID 只保存一次，避免不断覆盖导致网页闪烁
    if os.path.exists(path):
        return True
    return cv2.imwrite(
        path,
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
def build_shot_crop(img, c1, c2, track_id, label='',
                    scale_x=1.0, scale_y=1.0):
    """只生成抓拍图，不立即保存"""

    if img is None or img.size == 0:
        return None

    img_h, img_w = img.shape[:2]

    # 半分辨率检测框映射回原始高清帧
    x1 = int(c1[0] * scale_x)
    y1 = int(c1[1] * scale_y)
    x2 = int(c2[0] * scale_x)
    y2 = int(c2[1] * scale_y)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # 你现在已经验证过比较合适的 12% 留白
    pad_x = max(4, int(box_w * 0.12))
    pad_y = max(4, int(box_h * 0.12))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    # 小图放大
    crop_h, crop_w = crop.shape[:2]

    if crop_w < 240:
        ratio = 240.0 / crop_w
        new_h = max(1, int(crop_h * ratio))

        crop = cv2.resize(
            crop,
            (240, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    # 添加车型 + ID
    title = '{} ID-{}'.format(label, track_id)

    cv2.rectangle(
        crop,
        (0, 0),
        (min(crop.shape[1], 180), 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        crop,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return crop


def save_shot_crop(crop, track_id):
    """把已经选好的最佳帧保存下来"""

    if crop is None or crop.size == 0:
        return False

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    save_path = os.path.join(
        project_root,
        'src',
        'static',
        'save_img'
    )

    os.makedirs(save_path, exist_ok=True)

    path = os.path.join(
        save_path,
        '{}.jpg'.format(track_id)
    )

    # 一个 ID 只保存一次
    if os.path.exists(path):
        return True

    return cv2.imwrite(
        path,
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
class Detector_sg:

    def __init__(self):
        """
        初始化相关配置及其权重
        """
        base_path = os.path.dirname(os.getcwd())
        config_path = base_path + r'\web_final-master\src\templates\detector_sg_config.json'
        with open(config_path, 'rb') as _config:
            config = json.load(_config)
        self.img_size = config['img_size']  # 大小
        self.threshold = config['threshold']  # 置信度
        self.stride = config['stride']  # 步长
        self.weights = "yolo_weight/" + config['weight']  # 权重文件
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.float()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB
        img = np.ascontiguousarray(img)  # 地址连续化
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ['sg']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, 'accident', conf))

        return boxes
