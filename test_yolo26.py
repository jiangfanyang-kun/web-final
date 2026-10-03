from ultralytics import YOLO
import torch
import cv2


print('================================')
print('YOLO26 TEST')
print('Torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        'GPU:',
        torch.cuda.get_device_name(0)
    )

print('================================')


# 加载 YOLO26n
# 第一次运行会自动下载 yolo26n.pt
model = YOLO('yolo26n.pt')


# 使用项目已有的 traffic.mp4 测试
cap = cv2.VideoCapture('traffic.mp4')

if not cap.isOpened():
    print('traffic.mp4 打开失败')
    raise SystemExit


ok, frame = cap.read()

cap.release()


if not ok or frame is None:
    print('无法读取 traffic.mp4 第一帧')
    raise SystemExit


# GPU推理
results = model.predict(
    source=frame,
    device=0,
    verbose=False
)


result = results[0]


print(
    '检测目标数量:',
    len(result.boxes)
)


for box in result.boxes:

    cls_id = int(
        box.cls[0]
    )

    conf = float(
        box.conf[0]
    )

    xyxy = (
        box.xyxy[0]
        .cpu()
        .tolist()
    )

    label = result.names[
        cls_id
    ]

    print(
        label,
        'conf =',
        round(conf, 3),
        'xyxy =',
        [
            round(v, 1)
            for v in xyxy
        ]
    )


print('================================')
print('YOLO26 推理成功')
print('================================')