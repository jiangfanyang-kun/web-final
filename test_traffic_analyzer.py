# -*- coding: utf-8 -*-

import os
import sys
import cv2


# =========================================================
# 项目路径
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(
        0,
        SRC_PATH
    )


# =========================================================
# 导入
# =========================================================

from tools.tracker_yolo26 import TrafficTracker26
from tools.traffic_analyzer import TrafficAnalyzer


# =========================================================
# 视频
# =========================================================

VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    "traffic.mp4"
)


# =========================================================
# 主程序
# =========================================================

def main():

    print("=" * 60)
    print("YOLO26 + TrackTrack + TrafficAnalyzer 测试")
    print("=" * 60)

    if not os.path.exists(VIDEO_PATH):

        print("[ERROR] 找不到视频：")
        print(VIDEO_PATH)

        return

    # -----------------------------------------------------
    # 打开视频
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print("[ERROR] 视频打开失败")

        return

    print("[VIDEO] 视频打开成功")
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        source_fps = 30.0

    print("[VIDEO] 原始视频 FPS:", source_fps)

    # -----------------------------------------------------
    # 创建 TrackTrack
    # -----------------------------------------------------

    traffic_tracker = TrafficTracker26()

    # -----------------------------------------------------
    # 创建交通分析器
    # -----------------------------------------------------

    analyzer = TrafficAnalyzer()

    print("[ANALYZER] TrafficAnalyzer 创建成功")

    # -----------------------------------------------------
    # 测试帧数
    # -----------------------------------------------------

    max_frames = 100

    frame_count = 0

    try:

        while frame_count < max_frames:

            ret, frame = cap.read()

            if not ret:

                print("[VIDEO] 视频读取结束")

                break

            frame_count += 1

            # -------------------------------------------------
            # YOLO26 + TrackTrack
            # -------------------------------------------------

            tracks = traffic_tracker.track(
                frame
            )

            # -------------------------------------------------
            # TrafficAnalyzer
            # -------------------------------------------------
            # 使用视频时间，而不是电脑实际处理时间
            # 优先使用视频本身的时间戳
            video_time_ms = cap.get(
                cv2.CAP_PROP_POS_MSEC
            )

            if video_time_ms > 0:
                video_time = video_time_ms / 1000.0
            else:
                video_time = frame_count / source_fps

            result = analyzer.update(
                tracks,
                timestamp=video_time,
                frame_shape=frame.shape
            )

            # -------------------------------------------------
            # 每 10 帧打印一次
            # -------------------------------------------------

            if (
                frame_count == 1
                or frame_count % 10 == 0
            ):

                print(
                    "\nFrame:",
                    frame_count
                )

                print(
                    "车辆数量:",
                    result["vehicle_count"]
                )

                print(
                    "车型统计:",
                    result["class_counts"]
                )

                print(
                    "平均速度(px/s):",
                    result["average_speed_px_s"]
                )

                print(
                    "移动车辆:",
                    result["moving_vehicle_count"]
                )

                print(
                    "静止车辆:",
                    result["stopped_vehicle_count"]
                )

                print(
                    "静止比例:",
                    result["stopped_ratio"]
                )

                print(
                    "画面占用率:",
                    result["occupancy_ratio"]
                )

                print(
                    "当前Track ID:",
                    result["track_ids"]
                )

    finally:

        cap.release()

    # =====================================================
    # 最终结果
    # =====================================================

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

    print(
        "处理帧数:",
        frame_count
    )

    print(
        "当前活跃车辆:",
        analyzer.get_latest()["vehicle_count"]
    )

    print(
        "当前所有Track:"
    )

    for track in analyzer.get_all_tracks():

        print(
            "  ID={}".format(
                track["track_id"]
            ),
            "类型={}".format(
                track["label"]
            ),
            "速度={:.2f}px/s".format(
                track["speed_px_s"]
            ),
            "累计距离={:.2f}px".format(
                track["total_distance_px"]
            )
        )

    print("=" * 60)


if __name__ == "__main__":
    main()