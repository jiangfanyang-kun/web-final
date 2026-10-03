import cv2

from tools.tracker_yolo26 import TrafficTracker26


print('======================================')
print('YOLO26 + TrackTrack TEST')
print('======================================')


tracker26 = TrafficTracker26()


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


    tracks = tracker26.track(
        frame
    )


    print(
        'Frame:',
        frame_index,
        '| Tracks:',
        len(tracks)
    )


    for item in tracks:

        (
            x1,
            y1,
            x2,
            y2,
            label,
            track_id,
            confidence
        ) = item


        print(
            '   ID:',
            track_id,
            '|',
            label,
            '| conf:',
            round(confidence, 3),
            '|',
            (x1, y1, x2, y2)
        )


    # 先测试100帧
    if frame_index >= 100:
        break


cap.release()


print('======================================')
print('YOLO26 + TrackTrack 测试完成')
print('======================================')