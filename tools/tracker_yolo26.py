# encoding: utf-8

from ultralytics import YOLO
import torch


class TrafficTracker26:
    """
    YOLO26 + TrackTrack 统一交通感知器

    它直接完成：
        目标检测
        +
        多目标跟踪

    对外统一返回：
        [
            x1,
            y1,
            x2,
            y2,
            label,
            track_id,
            confidence
        ]

    后面的停车、逆行、拥堵、TTC等模块
    只依赖这个统一格式，
    不再依赖 YOLO / TrackTrack 内部对象。
    """

    def __init__(
            self,
            weights='yolo26n.pt',
            conf=0.20
    ):

        self.weights = weights
        self.conf = conf

        if torch.cuda.is_available():

            self.device = 0

            print(
                '[TRAFFIC26] GPU:',
                torch.cuda.get_device_name(0)
            )

        else:

            self.device = 'cpu'

            print(
                '[TRAFFIC26] 使用 CPU'
            )


        print(
            '[TRAFFIC26] 加载 YOLO26:',
            self.weights
        )

        self.model = YOLO(
            self.weights
        )

        print(
            '[TRAFFIC26] YOLO26 加载完成'
        )

        print(
            '[TRAFFIC26] Tracker: TrackTrack'
        )


        # COCO交通相关类别
        self.target_classes = {
            'person',
            'bicycle',
            'car',
            'motorcycle',
            'bus',
            'truck'
        }


    def track(self, frame):

        if frame is None:

            return []


        # ==================================================
        # YOLO26 + TrackTrack
        #
        # persist=True：
        # 告诉跟踪器当前帧与上一帧属于同一个连续视频
        # ==================================================

        results = self.model.track(
            source=frame,
            persist=True,
            tracker='tracktrack.yaml',
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


        # TrackTrack尚未产生ID
        if boxes.id is None:

            return []


        track_results = []


        # ==================================================
        # 转成我们项目自己的统一格式
        # ==================================================

        for box in boxes:

            # 某些尚未确认的目标可能没有ID
            if box.id is None:

                continue


            cls_id = int(
                box.cls[0].item()
            )

            label = result.names[
                cls_id
            ]


            # 过滤无关类别
            if label not in self.target_classes:

                continue


            track_id = int(
                box.id[0].item()
            )


            confidence = float(
                box.conf[0].item()
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


            track_results.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    label,
                    track_id,
                    confidence
                ]
            )


        return track_results