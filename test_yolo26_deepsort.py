import cv2

from tools.detector_yolo26 import Detector26
from tools import tracker


print('======================================')
print('YOLO26 + DeepSORT TEST')
print('======================================')


# 初始化 YOLO26
detector = Detector26()


# DeepSORT 轨迹历史
dict_box = {}


# 打开测试视频
cap = cv2.VideoCapture(
    'traffic.mp4'
)


if not cap.isOpened():

    print('traffic.mp4 打开失败')

    raise SystemExit


frame_index = 0


while True:

    ok, frame = cap.read()

    if not ok or frame is None:
        break


    frame_index += 1


    # ======================================
    # 1. YOLO26 检测
    # ======================================

    bboxes = detector.detect(
        frame
    )


    # ======================================
    # 2. DeepSORT 跟踪
    # ======================================

    if len(bboxes) > 0:

        tracks = tracker.update(
            bboxes,
            frame,
            dict_box
        )

    else:

        tracks = []


    # ======================================
    # 3. 打印跟踪结果
    # ======================================

    print(
        'Frame:',
        frame_index,
        '| YOLO:',
        len(bboxes),
        '| Tracks:',
        len(tracks)
    )


    for item in tracks:

        x1, y1, x2, y2, label, track_id, _ = item

        print(
            '   ID:',
            track_id,
            '|',
            label,
            '|',
            (x1, y1, x2, y2)
        )


    # ======================================
    # 只测试前 50 帧
    # ======================================

    if frame_index >= 50:
        break


cap.release()


print('======================================')
print('YOLO26 + DeepSORT 测试结束')
print('======================================')