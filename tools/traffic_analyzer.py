# -*- coding: utf-8 -*-

"""
交通流量分析器

作用：
1. 接收 YOLO26 + TrackTrack 输出的车辆轨迹
2. 维护每一个 Track ID 的历史状态
3. 计算车辆运动速度、累计距离、停留时间
4. 统计当前车辆数量、车型数量、静止车辆比例等
5. 保存最近一段时间的交通统计历史

注意：
当前速度单位为 px/s（像素/秒）。
如果以后需要 km/h，需要进行道路尺度标定。
"""

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class TrackState:
    """
    单个车辆 Track 的历史状态。
    """

    track_id: int

    label: str

    first_seen: float

    last_seen: float

    last_center: tuple

    last_bbox: tuple

    last_confidence: float = 0.0

    speed_px_s: float = 0.0

    raw_speed_px_s: float = 0.0

    total_distance_px: float = 0.0

    stationary_since: float = None

    stationary_duration: float = 0.0

    seen_frames: int = 1

    history: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )


class TrafficAnalyzer:
    """
    交通分析核心类。

    输入格式：

    [
        [
            x1,
            y1,
            x2,
            y2,
            label,
            track_id,
            confidence
        ],
        ...
    ]

    与当前 TrafficTracker26.track() 的输出格式兼容。
    """

    def __init__(
        self,
        history_size=300,
        track_history_size=30,
        max_missing_seconds=2.0,
        stationary_speed_px_s=8.0,
        speed_smoothing=0.35
    ):
        """
        参数：

        history_size:
            保存多少帧的整体交通统计。

        track_history_size:
            每辆车保存多少个历史位置。

        max_missing_seconds:
            Track 多久没有再次出现后，从分析器中清除。

        stationary_speed_px_s:
            低于这个速度时，认为车辆处于“基本静止”。

        speed_smoothing:
            速度 EMA 平滑系数。
            越大越灵敏，越小越平滑。
        """

        self.history_size = int(history_size)

        self.track_history_size = int(track_history_size)

        self.max_missing_seconds = float(
            max_missing_seconds
        )

        self.stationary_speed_px_s = float(
            stationary_speed_px_s
        )

        self.speed_smoothing = float(
            speed_smoothing
        )

        # 当前所有活跃 Track
        self.tracks = {}

        # 整体交通历史
        self.history = deque(
            maxlen=self.history_size
        )

        # 最近一次统计结果
        self.latest = {
            "timestamp": None,
            "vehicle_count": 0,
            "class_counts": {},
            "average_speed_px_s": 0.0,
            "moving_vehicle_count": 0,
            "stopped_vehicle_count": 0,
            "stopped_ratio": 0.0,
            "occupancy_ratio": 0.0,
            "track_ids": [],
        }

    # =========================================================
    # 基础工具
    # =========================================================

    @staticmethod
    def _center(x1, y1, x2, y2):
        """
        计算 bbox 中心点。
        """

        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0

        return cx, cy

    @staticmethod
    def _distance(point_a, point_b):
        """
        计算两个二维点之间的欧氏距离。
        """

        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]

        return math.sqrt(
            dx * dx + dy * dy
        )

    @staticmethod
    def _bbox_area(x1, y1, x2, y2):
        """
        计算 bbox 面积。
        """

        width = max(
            0.0,
            float(x2) - float(x1)
        )

        height = max(
            0.0,
            float(y2) - float(y1)
        )

        return width * height

    # =========================================================
    # 更新单辆车
    # =========================================================

    def _update_track(
        self,
        bbox,
        label,
        track_id,
        confidence,
        timestamp
    ):
        """
        更新一个 Track。
        """

        x1, y1, x2, y2 = bbox

        center = self._center(
            x1,
            y1,
            x2,
            y2
        )

        track_id = int(track_id)

        confidence = float(confidence)

        # -----------------------------------------------------
        # 新 Track
        # -----------------------------------------------------

        if track_id not in self.tracks:

            state = TrackState(
                track_id=track_id,
                label=str(label),
                first_seen=timestamp,
                last_seen=timestamp,
                last_center=center,
                last_bbox=(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ),
                last_confidence=confidence
            )

            state.history.append(
                (
                    timestamp,
                    center[0],
                    center[1]
                )
            )

            self.tracks[track_id] = state

            return state

        # -----------------------------------------------------
        # 已存在 Track
        # -----------------------------------------------------

        state = self.tracks[track_id]

        dt = timestamp - state.last_seen

        # 防止时间间隔为 0 或异常
        if dt <= 0:
            dt = 1e-6

        distance = self._distance(
            center,
            state.last_center
        )

        raw_speed = distance / dt

        # -----------------------------------------------------
        # EMA 平滑速度
        # -----------------------------------------------------

        alpha = self.speed_smoothing

        if state.seen_frames <= 1:
            smooth_speed = raw_speed
        else:
            smooth_speed = (
                alpha * raw_speed
                +
                (1.0 - alpha) * state.speed_px_s
            )

        # -----------------------------------------------------
        # 更新状态
        # -----------------------------------------------------

        state.label = str(label)

        state.last_seen = timestamp

        state.last_center = center

        state.last_bbox = (
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        )

        state.last_confidence = confidence

        state.raw_speed_px_s = raw_speed

        state.speed_px_s = smooth_speed

        state.total_distance_px += distance

        state.seen_frames += 1

        state.history.append(
            (
                timestamp,
                center[0],
                center[1]
            )
        )

        # -----------------------------------------------------
        # 判断是否静止
        # -----------------------------------------------------

        if state.speed_px_s <= self.stationary_speed_px_s:

            if state.stationary_since is None:
                state.stationary_since = state.last_seen

            state.stationary_duration = (
                timestamp
                -
                state.stationary_since
            )

        else:

            state.stationary_since = None

            state.stationary_duration = 0.0

        return state

    # =========================================================
    # 清理长时间消失的 Track
    # =========================================================

    def _cleanup_tracks(self, timestamp):
        """
        清理已经长时间没有出现的 Track。
        """

        expired_ids = []

        for track_id, state in self.tracks.items():

            missing_time = (
                timestamp
                -
                state.last_seen
            )

            if missing_time > self.max_missing_seconds:
                expired_ids.append(track_id)

        for track_id in expired_ids:

            del self.tracks[track_id]

    # =========================================================
    # 主更新函数
    # =========================================================

    def update(
        self,
        tracks,
        timestamp=None,
        frame_shape=None
    ):
        """
        更新当前帧。

        参数：

        tracks:
            TrafficTracker26.track() 返回的数据。

        timestamp:
            时间戳。
            不传时自动使用 monotonic()。

        frame_shape:
            OpenCV frame.shape。

            例如：

                frame.shape
                # (720, 1280, 3)

            用于估算车辆 bbox 占画面的比例。

        返回：

        dict
            当前交通统计结果。
        """

        if timestamp is None:
            timestamp = time.monotonic()

        timestamp = float(timestamp)

        current_ids = set()

        current_boxes = []

        class_counts = Counter()

        # -----------------------------------------------------
        # 更新当前所有车辆
        # -----------------------------------------------------

        for item in tracks:

            if item is None:
                continue

            if len(item) < 7:
                continue

            try:

                x1, y1, x2, y2 = map(
                    float,
                    item[:4]
                )

                label = str(item[4])

                track_id = int(item[5])

                confidence = float(item[6])

            except (
                TypeError,
                ValueError,
                IndexError
            ):
                continue

            if track_id < 0:
                continue

            self._update_track(
                bbox=(
                    x1,
                    y1,
                    x2,
                    y2
                ),
                label=label,
                track_id=track_id,
                confidence=confidence,
                timestamp=timestamp
            )

            current_ids.add(track_id)

            current_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            class_counts[label] += 1

        # -----------------------------------------------------
        # 清理已经消失的 Track
        # -----------------------------------------------------

        self._cleanup_tracks(
            timestamp
        )

        # -----------------------------------------------------
        # 当前活跃 Track
        # -----------------------------------------------------

        active_states = []

        for track_id in current_ids:

            state = self.tracks.get(
                track_id
            )

            if state is not None:
                active_states.append(state)

        # -----------------------------------------------------
        # 车辆数量
        # -----------------------------------------------------

        vehicle_count = len(
            active_states
        )

        # -----------------------------------------------------
        # 平均速度
        # -----------------------------------------------------

        if vehicle_count > 0:

            average_speed = sum(
                state.speed_px_s
                for state in active_states
            ) / vehicle_count

        else:

            average_speed = 0.0

        # -----------------------------------------------------
        # 移动车辆 / 静止车辆
        # -----------------------------------------------------

        stopped_count = 0

        moving_count = 0

        for state in active_states:

            if (
                state.speed_px_s
                <=
                self.stationary_speed_px_s
            ):
                stopped_count += 1
            else:
                moving_count += 1

        # -----------------------------------------------------
        # 静止比例
        # -----------------------------------------------------

        if vehicle_count > 0:

            stopped_ratio = (
                stopped_count
                /
                vehicle_count
            )

        else:

            stopped_ratio = 0.0

        # -----------------------------------------------------
        # bbox 占画面比例
        # -----------------------------------------------------

        occupancy_ratio = 0.0

        if (
            frame_shape is not None
            and len(frame_shape) >= 2
        ):

            frame_height = float(
                frame_shape[0]
            )

            frame_width = float(
                frame_shape[1]
            )

            frame_area = (
                frame_width
                *
                frame_height
            )

            if frame_area > 0:

                bbox_area = sum(
                    self._bbox_area(
                        x1,
                        y1,
                        x2,
                        y2
                    )
                    for (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                    in current_boxes
                )

                occupancy_ratio = (
                    bbox_area
                    /
                    frame_area
                )

        # -----------------------------------------------------
        # 当前统计结果
        # -----------------------------------------------------

        result = {
            "timestamp": timestamp,

            "vehicle_count": vehicle_count,

            "class_counts": dict(
                class_counts
            ),

            "average_speed_px_s": round(
                average_speed,
                2
            ),

            "moving_vehicle_count": moving_count,

            "stopped_vehicle_count": stopped_count,

            "stopped_ratio": round(
                stopped_ratio,
                4
            ),

            "occupancy_ratio": round(
                occupancy_ratio,
                4
            ),

            "track_ids": sorted(
                list(current_ids)
            ),
        }

        # -----------------------------------------------------
        # 保存历史
        # -----------------------------------------------------

        self.history.append(
            result.copy()
        )

        self.latest = result

        return result

    # =========================================================
    # 获取单辆车信息
    # =========================================================

    def get_track(self, track_id):
        """
        获取指定 Track ID 的详细信息。
        """

        state = self.tracks.get(
            int(track_id)
        )

        if state is None:
            return None

        return {
            "track_id": state.track_id,

            "label": state.label,

            "speed_px_s": round(
                state.speed_px_s,
                2
            ),

            "raw_speed_px_s": round(
                state.raw_speed_px_s,
                2
            ),

            "total_distance_px": round(
                state.total_distance_px,
                2
            ),

            "stationary_duration": round(
                state.stationary_duration,
                2
            ),

            "seen_frames": state.seen_frames,

            "confidence": round(
                state.last_confidence,
                4
            ),

            "bbox": state.last_bbox,

            "center": state.last_center,

            "first_seen": state.first_seen,

            "last_seen": state.last_seen,
        }

    # =========================================================
    # 获取当前所有车辆
    # =========================================================

    def get_all_tracks(self):
        """
        获取当前所有活跃车辆信息。
        """

        result = []

        for track_id in sorted(
            self.tracks.keys()
        ):

            info = self.get_track(
                track_id
            )

            if info is not None:
                result.append(info)

        return result

    # =========================================================
    # 获取最近历史
    # =========================================================

    def get_history(self, limit=None):
        """
        获取交通统计历史。
        """

        data = list(
            self.history
        )

        if limit is not None:

            limit = int(limit)

            if limit > 0:
                data = data[-limit:]

        return data

    # =========================================================
    # 获取最新统计
    # =========================================================

    def get_latest(self):
        """
        获取最近一次统计结果。
        """

        return self.latest.copy()

    # =========================================================
    # 重置
    # =========================================================

    def reset(self):
        """
        清空所有分析数据。
        """

        self.tracks.clear()

        self.history.clear()

        self.latest = {
            "timestamp": None,
            "vehicle_count": 0,
            "class_counts": {},
            "average_speed_px_s": 0.0,
            "moving_vehicle_count": 0,
            "stopped_vehicle_count": 0,
            "stopped_ratio": 0.0,
            "occupancy_ratio": 0.0,
            "track_ids": [],
        }