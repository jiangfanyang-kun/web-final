import cv2

from tools.detector_yolo26 import Detector26


print('==============================')
print('Detector26 compatibility test')
print('==============================')


detector = Detector26()


cap = cv2.VideoCapture(
    'traffic.mp4'
)


if not cap.isOpened():

    print('traffic.mp4 打开失败')

    raise SystemExit


ok, frame = cap.read()

cap.release()


if not ok or frame is None:

    print('读取视频第一帧失败')

    raise SystemExit


bboxes = detector.detect(
    frame
)


print(
    '检测数量:',
    len(bboxes)
)


for bbox in bboxes:

    print(
        bbox
    )


print('==============================')


if len(bboxes) > 0:

    print(
        '第一个 bbox 长度:',
        len(bboxes[0])
    )


print(
    'Detector26 测试完成'
)