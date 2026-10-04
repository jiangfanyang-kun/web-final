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
    # 逆行开始时间
    reverse_since: float = None

    # 连续逆行持续时间
    reverse_duration: float = 0.0
    seen_frames: int = 1

    history: deque = field(
        default_factory=deque
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
            state.history = deque(
                maxlen=self.track_history_size
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

        # =====================================================
        # 交通状态中认定为“车辆”的类别
        #
        # person 仍然保留在 tracks/current_ids 中，
        # 因为后面的人车冲突预警还需要行人轨迹。
        # 但 person 不参与车辆数量、车辆平均速度、
        # 静止比例和车辆画面占用率计算。
        # =====================================================

        vehicle_labels = {
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle"
        }

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

            # 所有目标 ID 都保留
            # person 后面的人车冲突检测还要使用
            current_ids.add(track_id)

            # 只有车辆 bbox 才进入交通占用率计算
            if label in vehicle_labels:
                current_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

            # 类别统计仍然保留所有类别
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

            # 交通状态只统计车辆
            # person 不进入 vehicle_count / speed /
            # stopped_ratio 等车辆指标
            if (
                    state is not None
                    and
                    state.label in vehicle_labels
            ):
                active_states.append(
                    state
                )

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
    # 获取车辆运动方向
    # =========================================================

    def get_motion_direction(
        self,
        track_id,
        lookback_seconds=1.0,
        min_distance_px=15.0
    ):
        """
        根据同一个 Track ID 最近一段时间的轨迹，
        判断车辆主要运动方向。

        返回方向：
            left
            right
            up
            down
            stationary
            unknown
        """

        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return {
                "direction": "unknown",
                "dx": 0.0,
                "dy": 0.0,
                "distance_px": 0.0
            }

        state = self.tracks.get(track_id)

        if state is None:
            return {
                "direction": "unknown",
                "dx": 0.0,
                "dy": 0.0,
                "distance_px": 0.0
            }

        history = list(state.history)

        if len(history) < 2:
            return {
                "direction": "unknown",
                "dx": 0.0,
                "dy": 0.0,
                "distance_px": 0.0
            }

        newest = history[-1]

        newest_time = float(newest[0])

        # 找最近约 1 秒以前的轨迹点
        old_point = history[0]

        for point in reversed(history[:-1]):

            point_time = float(point[0])

            if (
                newest_time - point_time
                >= lookback_seconds
            ):
                old_point = point
                break

        dx = (
            float(newest[1])
            -
            float(old_point[1])
        )

        dy = (
            float(newest[2])
            -
            float(old_point[2])
        )

        distance_px = math.sqrt(
            dx * dx + dy * dy
        )

        # 位移太小，不判断方向
        if distance_px < min_distance_px:
            direction = "stationary"

        # 水平方向变化更明显
        elif abs(dx) >= abs(dy):

            if dx > 0:
                direction = "right"
            else:
                direction = "left"

        # 垂直方向变化更明显
        else:

            if dy > 0:
                direction = "down"
            else:
                direction = "up"

        return {
            "direction": direction,

            "dx": round(
                dx,
                2
            ),

            "dy": round(
                dy,
                2
            ),

            "distance_px": round(
                distance_px,
                2
            )
        }
    # =========================================================
    # 逆行判定
    # =========================================================

    def check_reverse_direction(
        self,
        track_id,
        expected_direction=None,
        min_reverse_seconds=1.5
    ):
        """
        判断指定车辆是否持续逆行。

        expected_direction:
            right
            left
            up
            down

        例如：
            正常方向为 right
            当前车辆持续向 left
            >= min_reverse_seconds
            则判定为逆行。

        expected_direction 为 None 时，
        表示当前道路尚未配置正常行驶方向，
        不触发逆行报警。
        """

        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return {
                "configured": False,
                "is_reverse": False,
                "reverse_duration": 0.0,
                "expected_direction": expected_direction,
                "actual_direction": "unknown"
            }

        state = self.tracks.get(track_id)

        if state is None:
            return {
                "configured": False,
                "is_reverse": False,
                "reverse_duration": 0.0,
                "expected_direction": expected_direction,
                "actual_direction": "unknown"
            }

        valid_directions = {
            "left",
            "right",
            "up",
            "down"
        }

        # 道路正常方向还没有配置
        if expected_direction not in valid_directions:

            state.reverse_since = None
            state.reverse_duration = 0.0

            return {
                "configured": False,
                "is_reverse": False,
                "reverse_duration": 0.0,
                "expected_direction": expected_direction,
                "actual_direction": "unknown"
            }

        direction_info = self.get_motion_direction(
            track_id
        )

        actual_direction = direction_info[
            "direction"
        ]

        opposite_direction = {
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up"
        }

        reverse_direction = opposite_direction[
            expected_direction
        ]

        # 当前运动方向确实与正常方向完全相反
        if actual_direction == reverse_direction:

            if state.reverse_since is None:
                state.reverse_since = state.last_seen

            state.reverse_duration = max(
                0.0,
                state.last_seen
                -
                state.reverse_since
            )

        else:

            # 方向恢复正常、未知或暂时静止
            # 当前连续逆行计时清零
            state.reverse_since = None
            state.reverse_duration = 0.0

        is_reverse = (
            state.reverse_duration
            >=
            float(min_reverse_seconds)
        )

        return {
            "configured": True,

            "is_reverse": is_reverse,

            "reverse_duration": round(
                state.reverse_duration,
                2
            ),

            "expected_direction": expected_direction,

            "actual_direction": actual_direction,

            "reverse_direction": reverse_direction
        }
    # =========================================================
    # 获取车辆速度向量
    # =========================================================

    def get_velocity_vector(
        self,
        track_id,
        lookback_seconds=0.8
    ):
        """
        根据最近一段轨迹计算车辆速度向量。

        返回：
            vx: X方向速度 px/s
            vy: Y方向速度 px/s
            speed_px_s: 合速度 px/s
        """

        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return {
                "vx": 0.0,
                "vy": 0.0,
                "speed_px_s": 0.0,
                "valid": False
            }

        state = self.tracks.get(track_id)

        if state is None:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "speed_px_s": 0.0,
                "valid": False
            }

        history = list(state.history)

        if len(history) < 2:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "speed_px_s": 0.0,
                "valid": False
            }

        newest = history[-1]

        newest_time = float(newest[0])

        old_point = history[0]

        # 尽量取约 lookback_seconds 以前的轨迹点
        for point in reversed(history[:-1]):

            if (
                newest_time
                -
                float(point[0])
                >= lookback_seconds
            ):
                old_point = point
                break

        dt = (
            newest_time
            -
            float(old_point[0])
        )

        if dt <= 0.05:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "speed_px_s": 0.0,
                "valid": False
            }

        dx = (
            float(newest[1])
            -
            float(old_point[1])
        )

        dy = (
            float(newest[2])
            -
            float(old_point[2])
        )

        vx = dx / dt
        vy = dy / dt

        speed_px_s = math.sqrt(
            vx * vx
            +
            vy * vy
        )

        return {
            "vx": round(vx, 2),
            "vy": round(vy, 2),
            "speed_px_s": round(
                speed_px_s,
                2
            ),
            "valid": True
        }


    # =========================================================
    # TTC 碰撞风险分析
    # =========================================================

    def get_ttc_risks(
        self,
        max_ttc_seconds=3.0,
        collision_distance_px=50.0,
        min_closing_speed_px_s=10.0,
        lookback_seconds=0.8
    ):
        """
        根据两辆车的位置和速度向量，
        预测在匀速运动假设下是否可能接近碰撞。

        当前全部基于像素坐标：

            距离：px
            速度：px/s
            TTC：s

        后续做道路标定后再升级真实距离模型。
        """

        risks = []

        # 只分析当前这一帧真正存在的 Track
        active_ids = self.latest.get(
            "track_ids",
            []
        )

        # TTC 这里只分析机动车/非机动车之间的风险
        vehicle_labels = {
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle"
        }

        active_ids = [
            int(track_id)
            for track_id in active_ids
            if (
                    int(track_id) in self.tracks
                    and
                    self.tracks[
                        int(track_id)
                    ].label in vehicle_labels
            )
        ]

        # 两两组合
        for i in range(len(active_ids)):

            for j in range(
                i + 1,
                len(active_ids)
            ):

                id_a = active_ids[i]
                id_b = active_ids[j]

                state_a = self.tracks.get(id_a)
                state_b = self.tracks.get(id_b)

                if (
                    state_a is None
                    or state_b is None
                ):
                    continue

                velocity_a = (
                    self.get_velocity_vector(
                        id_a,
                        lookback_seconds
                    )
                )

                velocity_b = (
                    self.get_velocity_vector(
                        id_b,
                        lookback_seconds
                    )
                )

                if (
                    not velocity_a["valid"]
                    or not velocity_b["valid"]
                ):
                    continue

                # ---------------------------------------------
                # 当前相对位置
                # ---------------------------------------------

                ax, ay = state_a.last_center
                bx, by = state_b.last_center

                rx = bx - ax
                ry = by - ay

                current_distance = math.sqrt(
                    rx * rx
                    +
                    ry * ry
                )

                if current_distance <= 1e-6:
                    continue

                # ---------------------------------------------
                # 相对速度
                # ---------------------------------------------

                rvx = (
                    velocity_b["vx"]
                    -
                    velocity_a["vx"]
                )

                rvy = (
                    velocity_b["vy"]
                    -
                    velocity_a["vy"]
                )

                relative_speed_sq = (
                    rvx * rvx
                    +
                    rvy * rvy
                )

                if relative_speed_sq <= 1e-6:
                    continue

                # ---------------------------------------------
                # 当前是否正在接近
                # ---------------------------------------------

                dot_r_v = (
                    rx * rvx
                    +
                    ry * rvy
                )

                closing_speed = (
                    -dot_r_v
                    /
                    current_distance
                )

                if (
                    closing_speed
                    <
                    min_closing_speed_px_s
                ):
                    continue

                # ---------------------------------------------
                # 到达最近点所需时间
                # ---------------------------------------------

                ttc = (
                    -dot_r_v
                    /
                    relative_speed_sq
                )

                if (
                    ttc <= 0.0
                    or
                    ttc > max_ttc_seconds
                ):
                    continue

                # ---------------------------------------------
                # 预测最近时刻的两车距离
                # ---------------------------------------------

                future_rx = (
                    rx
                    +
                    rvx * ttc
                )

                future_ry = (
                    ry
                    +
                    rvy * ttc
                )

                predicted_distance = math.sqrt(
                    future_rx * future_rx
                    +
                    future_ry * future_ry
                )

                # 路径虽然接近，但不会靠得足够近
                if (
                    predicted_distance
                    >
                    collision_distance_px
                ):
                    continue

                # ---------------------------------------------
                # 风险等级
                # ---------------------------------------------

                if ttc <= 1.0:
                    risk_level = "high"

                elif ttc <= 2.0:
                    risk_level = "medium"

                else:
                    risk_level = "low"

                risks.append(
                    {
                        "track_id_a": id_a,
                        "track_id_b": id_b,

                        "label_a": state_a.label,
                        "label_b": state_b.label,

                        "ttc_seconds": round(
                            ttc,
                            2
                        ),

                        "current_distance_px": round(
                            current_distance,
                            2
                        ),

                        "predicted_distance_px": round(
                            predicted_distance,
                            2
                        ),

                        "closing_speed_px_s": round(
                            closing_speed,
                            2
                        ),

                        "risk_level": risk_level
                    }
                )

        # 最危险的排最前面
        risks.sort(
            key=lambda item:
            item["ttc_seconds"]
        )

        return risks
    # =========================================================
    # 人车冲突风险分析
    # =========================================================

    def get_pedestrian_vehicle_risks(
        self,
        max_prediction_seconds=3.0,
        conflict_distance_px=60.0,
        min_closing_speed_px_s=5.0,
        lookback_seconds=0.8
    ):
        """
        根据行人与车辆的当前位置和运动方向，
        判断未来几秒内是否可能产生人车冲突。

        当前基于像素坐标进行预测：

            距离：px
            速度：px/s
            时间：s

        这是风险预警，不代表一定会发生碰撞。
        """

        risks = []

        active_ids = self.latest.get(
            "track_ids",
            []
        )

        person_ids = []
        vehicle_ids = []

        vehicle_labels = {
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle"
        }

        # -----------------------------------------------------
        # 把当前目标分成人和车辆
        # -----------------------------------------------------

        for track_id in active_ids:

            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                continue

            state = self.tracks.get(
                track_id
            )

            if state is None:
                continue

            if state.label == "person":

                person_ids.append(
                    track_id
                )

            elif state.label in vehicle_labels:

                vehicle_ids.append(
                    track_id
                )

        # -----------------------------------------------------
        # 每个行人与每辆车进行配对分析
        # -----------------------------------------------------

        for person_id in person_ids:

            person_state = self.tracks.get(
                person_id
            )

            person_velocity = (
                self.get_velocity_vector(
                    person_id,
                    lookback_seconds
                )
            )

            if not person_velocity[
                "valid"
            ]:
                continue

            for vehicle_id in vehicle_ids:

                vehicle_state = self.tracks.get(
                    vehicle_id
                )

                vehicle_velocity = (
                    self.get_velocity_vector(
                        vehicle_id,
                        lookback_seconds
                    )
                )

                if (
                    vehicle_state is None
                    or
                    not vehicle_velocity[
                        "valid"
                    ]
                ):
                    continue

                # ---------------------------------------------
                # 当前相对位置
                # ---------------------------------------------

                vehicle_x, vehicle_y = (
                    vehicle_state.last_center
                )

                person_x, person_y = (
                    person_state.last_center
                )

                rx = (
                    person_x
                    -
                    vehicle_x
                )

                ry = (
                    person_y
                    -
                    vehicle_y
                )

                current_distance = math.sqrt(
                    rx * rx
                    +
                    ry * ry
                )

                if current_distance <= 1e-6:
                    continue

                # ---------------------------------------------
                # 行人相对车辆的速度
                # ---------------------------------------------

                rvx = (
                    person_velocity["vx"]
                    -
                    vehicle_velocity["vx"]
                )

                rvy = (
                    person_velocity["vy"]
                    -
                    vehicle_velocity["vy"]
                )

                relative_speed_sq = (
                    rvx * rvx
                    +
                    rvy * rvy
                )

                if relative_speed_sq <= 1e-6:
                    continue

                dot_r_v = (
                    rx * rvx
                    +
                    ry * rvy
                )

                # ---------------------------------------------
                # 是否正在互相接近
                # ---------------------------------------------

                closing_speed = (
                    -dot_r_v
                    /
                    current_distance
                )

                if (
                    closing_speed
                    <
                    min_closing_speed_px_s
                ):
                    continue

                # ---------------------------------------------
                # 预测最近接近时刻
                # ---------------------------------------------

                conflict_time = (
                    -dot_r_v
                    /
                    relative_speed_sq
                )

                if (
                    conflict_time <= 0.0
                    or
                    conflict_time
                    >
                    max_prediction_seconds
                ):
                    continue

                # ---------------------------------------------
                # 预测最近距离
                # ---------------------------------------------

                future_rx = (
                    rx
                    +
                    rvx * conflict_time
                )

                future_ry = (
                    ry
                    +
                    rvy * conflict_time
                )

                predicted_distance = math.sqrt(
                    future_rx * future_rx
                    +
                    future_ry * future_ry
                )

                if (
                    predicted_distance
                    >
                    conflict_distance_px
                ):
                    continue

                # ---------------------------------------------
                # 风险等级
                # ---------------------------------------------

                if (
                    conflict_time <= 1.0
                    or
                    predicted_distance <= 20.0
                ):

                    risk_level = "high"

                elif conflict_time <= 2.0:

                    risk_level = "medium"

                else:

                    risk_level = "low"

                risks.append(
                    {
                        "person_id": person_id,

                        "vehicle_id": vehicle_id,

                        "vehicle_label":
                            vehicle_state.label,

                        "conflict_seconds": round(
                            conflict_time,
                            2
                        ),

                        "current_distance_px": round(
                            current_distance,
                            2
                        ),

                        "predicted_distance_px": round(
                            predicted_distance,
                            2
                        ),

                        "closing_speed_px_s": round(
                            closing_speed,
                            2
                        ),

                        "risk_level": risk_level
                    }
                )

        # 最危险的放前面
        risks.sort(
            key=lambda item:
            item["conflict_seconds"]
        )

        return risks
    # =========================================================
    # 跟车过近 / 安全距离风险分析
    # =========================================================

    def get_following_distance_risks(
        self,
        safe_gap_scale=2.0,
        lateral_scale=0.9,
        min_speed_px_s=8.0
    ):
        """
        判断同方向行驶的两辆车是否存在跟车过近风险。

        当前没有真实道路标定，因此使用：

            车辆中心距离
            +
            bbox 尺寸
            +
            行驶方向
            +
            相对速度

        构建像素坐标下的相对安全距离模型。

        后续完成摄像机标定后，
        可以替换成真实米制安全距离。
        """

        risks = []

        vehicle_labels = {
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle"
        }

        valid_directions = {
            "left",
            "right",
            "up",
            "down"
        }

        active_ids = self.latest.get(
            "track_ids",
            []
        )

        vehicle_ids = []

        # -----------------------------------------------------
        # 获取当前有效车辆
        # -----------------------------------------------------

        for track_id in active_ids:

            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                continue

            state = self.tracks.get(
                track_id
            )

            if state is None:
                continue

            if state.label not in vehicle_labels:
                continue

            direction_info = (
                self.get_motion_direction(
                    track_id
                )
            )

            direction = direction_info[
                "direction"
            ]

            if direction not in valid_directions:
                continue

            velocity = (
                self.get_velocity_vector(
                    track_id
                )
            )

            if not velocity["valid"]:
                continue

            vehicle_ids.append(
                {
                    "track_id": track_id,
                    "state": state,
                    "direction": direction,
                    "velocity": velocity
                }
            )

        # -----------------------------------------------------
        # 两两分析
        # -----------------------------------------------------

        for i in range(
            len(vehicle_ids)
        ):

            for j in range(
                i + 1,
                len(vehicle_ids)
            ):

                info_a = vehicle_ids[i]
                info_b = vehicle_ids[j]

                # 行驶方向必须一致
                if (
                    info_a["direction"]
                    !=
                    info_b["direction"]
                ):
                    continue

                direction = info_a[
                    "direction"
                ]

                state_a = info_a[
                    "state"
                ]

                state_b = info_b[
                    "state"
                ]

                ax, ay = state_a.last_center
                bx, by = state_b.last_center

                (
                    ax1,
                    ay1,
                    ax2,
                    ay2
                ) = state_a.last_bbox

                (
                    bx1,
                    by1,
                    bx2,
                    by2
                ) = state_b.last_bbox

                width_a = abs(
                    ax2 - ax1
                )

                height_a = abs(
                    ay2 - ay1
                )

                width_b = abs(
                    bx2 - bx1
                )

                height_b = abs(
                    by2 - by1
                )

                # ---------------------------------------------
                # 根据行驶方向判断前车 / 后车
                # ---------------------------------------------

                if direction == "right":

                    lateral_gap = abs(
                        ay - by
                    )

                    avg_long_size = (
                        width_a + width_b
                    ) / 2.0

                    avg_lateral_size = (
                        height_a + height_b
                    ) / 2.0

                    if ax <= bx:

                        rear_info = info_a
                        front_info = info_b

                        longitudinal_gap = (
                            bx - ax
                        )

                    else:

                        rear_info = info_b
                        front_info = info_a

                        longitudinal_gap = (
                            ax - bx
                        )


                elif direction == "left":

                    lateral_gap = abs(
                        ay - by
                    )

                    avg_long_size = (
                        width_a + width_b
                    ) / 2.0

                    avg_lateral_size = (
                        height_a + height_b
                    ) / 2.0

                    if ax >= bx:

                        rear_info = info_a
                        front_info = info_b

                        longitudinal_gap = (
                            ax - bx
                        )

                    else:

                        rear_info = info_b
                        front_info = info_a

                        longitudinal_gap = (
                            bx - ax
                        )


                elif direction == "down":

                    lateral_gap = abs(
                        ax - bx
                    )

                    avg_long_size = (
                        height_a + height_b
                    ) / 2.0

                    avg_lateral_size = (
                        width_a + width_b
                    ) / 2.0

                    if ay <= by:

                        rear_info = info_a
                        front_info = info_b

                        longitudinal_gap = (
                            by - ay
                        )

                    else:

                        rear_info = info_b
                        front_info = info_a

                        longitudinal_gap = (
                            ay - by
                        )


                else:
                    # direction == "up"

                    lateral_gap = abs(
                        ax - bx
                    )

                    avg_long_size = (
                        height_a + height_b
                    ) / 2.0

                    avg_lateral_size = (
                        width_a + width_b
                    ) / 2.0

                    if ay >= by:

                        rear_info = info_a
                        front_info = info_b

                        longitudinal_gap = (
                            ay - by
                        )

                    else:

                        rear_info = info_b
                        front_info = info_a

                        longitudinal_gap = (
                            by - ay
                        )

                # ---------------------------------------------
                # 判断是不是大致处于同一车道
                # ---------------------------------------------

                lateral_threshold = max(
                    20.0,
                    avg_lateral_size
                    *
                    lateral_scale
                )

                if (
                    lateral_gap
                    >
                    lateral_threshold
                ):
                    continue

                # ---------------------------------------------
                # 动态安全距离
                #
                # 根据车辆 bbox 大小进行归一化，
                # 比固定写死 50px 更适合透视画面。
                # ---------------------------------------------

                safe_distance = max(
                    35.0,
                    avg_long_size
                    *
                    safe_gap_scale
                )

                gap_ratio = (
                    longitudinal_gap
                    /
                    safe_distance
                )

                # 没进入安全距离范围
                if gap_ratio > 1.0:
                    continue

                rear_velocity = rear_info[
                    "velocity"
                ]

                front_velocity = front_info[
                    "velocity"
                ]

                rear_speed = rear_velocity[
                    "speed_px_s"
                ]

                front_speed = front_velocity[
                    "speed_px_s"
                ]

                # 后车基本没在运动，不作为跟车风险
                if rear_speed < min_speed_px_s:
                    continue

                closing_speed = (
                    rear_speed
                    -
                    front_speed
                )

                # ---------------------------------------------
                # 风险等级
                # ---------------------------------------------

                if (
                    gap_ratio <= 0.45
                    or
                    (
                        gap_ratio <= 0.65
                        and
                        closing_speed >= 20.0
                    )
                ):

                    risk_level = "high"

                elif (
                    gap_ratio <= 0.70
                    or
                    closing_speed >= 10.0
                ):

                    risk_level = "medium"

                else:

                    risk_level = "low"

                risks.append(
                    {
                        "rear_track_id":
                            rear_info["track_id"],

                        "front_track_id":
                            front_info["track_id"],

                        "direction":
                            direction,

                        "longitudinal_gap_px": round(
                            longitudinal_gap,
                            2
                        ),

                        "lateral_gap_px": round(
                            lateral_gap,
                            2
                        ),

                        "safe_distance_px": round(
                            safe_distance,
                            2
                        ),

                        "gap_ratio": round(
                            gap_ratio,
                            3
                        ),

                        "rear_speed_px_s": round(
                            rear_speed,
                            2
                        ),

                        "front_speed_px_s": round(
                            front_speed,
                            2
                        ),

                        "closing_speed_px_s": round(
                            closing_speed,
                            2
                        ),

                        "risk_level":
                            risk_level
                    }
                )

        # 间距比例越小越危险
        risks.sort(
            key=lambda item:
            item["gap_ratio"]
        )

        return risks
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
    # 获取最近时间窗口统计
    # =========================================================

    def get_window_summary(self, window_seconds=3.0):
        """
        对最近一段时间的交通数据做平均，
        减少单帧检测波动对交通状态的影响。
        """

        if not self.history:
            return {
                "sample_count": 0,
                "vehicle_count": 0.0,
                "average_speed_px_s": 0.0,
                "stopped_ratio": 0.0,
                "occupancy_ratio": 0.0,
            }

        latest_timestamp = self.history[-1].get(
            "timestamp"
        )

        if latest_timestamp is None:
            return {
                "sample_count": 0,
                "vehicle_count": 0.0,
                "average_speed_px_s": 0.0,
                "stopped_ratio": 0.0,
                "occupancy_ratio": 0.0,
            }

        cutoff_time = (
            float(latest_timestamp)
            -
            float(window_seconds)
        )

        recent = [
            item
            for item in self.history
            if (
                item.get("timestamp") is not None
                and
                float(item["timestamp"]) >= cutoff_time
            )
        ]

        if not recent:
            recent = [
                self.history[-1]
            ]

        # -----------------------------------------------------
        # 平均车辆数量
        # -----------------------------------------------------

        vehicle_count = (
            sum(
                item["vehicle_count"]
                for item in recent
            )
            /
            len(recent)
        )

        # -----------------------------------------------------
        # 只使用存在车辆的帧计算速度和静止比例
        # -----------------------------------------------------

        vehicle_frames = [
            item
            for item in recent
            if item["vehicle_count"] > 0
        ]

        if vehicle_frames:

            average_speed = (
                sum(
                    item["average_speed_px_s"]
                    for item in vehicle_frames
                )
                /
                len(vehicle_frames)
            )

            stopped_ratio = (
                sum(
                    item["stopped_ratio"]
                    for item in vehicle_frames
                )
                /
                len(vehicle_frames)
            )

        else:

            average_speed = 0.0
            stopped_ratio = 0.0

        # -----------------------------------------------------
        # 平均画面占用率
        # -----------------------------------------------------

        occupancy_ratio = (
            sum(
                item["occupancy_ratio"]
                for item in recent
            )
            /
            len(recent)
        )

        return {
            "sample_count": len(recent),

            "vehicle_count": round(
                vehicle_count,
                2
            ),

            "average_speed_px_s": round(
                average_speed,
                2
            ),

            "stopped_ratio": round(
                stopped_ratio,
                4
            ),

            "occupancy_ratio": round(
                occupancy_ratio,
                4
            ),
        }
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