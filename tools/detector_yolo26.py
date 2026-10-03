# encoding: utf-8

from ultralytics import YOLO
import torch


class Detector26:
    """
    YOLO26 交通目标检测器

    对外接口保持和旧 Detector 一致：

        detector.detect(image)

    返回：
        [
            (x1, y1, x2, y2, label, confidence),
            ...
        ]

    这样后面的 DeepSORT、流量统计、撞线、
    抓拍等旧代码不需要知道底层已经换成 YOLO26。
    """

    def __init__(
            self,
            weights='yolo26n.pt',
            conf=0.25
    ):

        self.weights = weights

        self.conf = conf

        # 自动选择 GPU / CPU
        if torch.cuda.is_available():

            self.device = 0

            print(
                '[YOLO26] GPU:',
                torch.cuda.get_device_name(0)
            )

        else:

            self.device = 'cpu'

            print(
                '[YOLO26] 使用 CPU'
            )


        print(
            '[YOLO26] 正在加载模型:',
            self.weights
        )

        self.model = YOLO(
            self.weights
        )

        print(
            '[YOLO26] 模型加载完成'
        )


        # ==================================================
        # 我们当前交通系统关心的目标
        # ==================================================

        self.target_classes = {
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'bus',
            'truck'
        }


    def detect(self, image):

        if image is None:

            return []


        # ==================================================
        # YOLO26 推理
        # ==================================================

        results = self.model.predict(
            source=image,
            conf=self.conf,
            device=self.device,
            verbose=False
        )


        if not results:

            return []


        result = results[0]

        boxes = result.boxes


        if boxes is None:

            return []


        detections = []


        # ==================================================
        # 转换为旧项目 Detector 的输出格式
        # ==================================================

        for box in boxes:

            cls_id = int(
                box.cls[0]
            )

            label = result.names[
                cls_id
            ]


            # 过滤非交通目标
            if label not in self.target_classes:

                continue


            confidence = float(
                box.conf[0]
            )


            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )


            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])


            detections.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    label,
                    confidence
                )
            )


        return detections