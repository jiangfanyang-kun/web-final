# author  : jerrylee
# data    : 2022/1/14
# time    : 20:30
# encoding: utf-8
import fileinput
import os
import threading

import numpy as np
from flask import Response, request
from flask import Flask
from flask import render_template
from tools.detector import (
    Detector_sg,
    shot_img,
    build_shot_crop,
    save_shot_crop
)

from tools.tracker_yolo26 import TrafficTracker26
from tools.traffic_analyzer import TrafficAnalyzer
import time
import torch
import json
import cv2
# ==========================================================
# GPU / OpenCV 性能优化
# ==========================================================

cv2.setUseOptimized(True)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
# 定义flask应用app入口
app = Flask(__name__)


# ==========================================================
# 项目路径配置
# ==========================================================

# web_main.py 位于 src 目录
# 往上一级就是整个项目根目录 web_final-master
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ==========================================================
# 两套功能使用各自独立的视频
# ==========================================================

# 普通交通检测视频
TRAFFIC_VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    'traffic.mp4'
)

# 事故检测视频
ACCIDENT_VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    'accident.mp4'
)
# =========================================================
# 当前正在使用的视频源
# =========================================================

# demo    = 本地典型场景视频
# network = 网络道路视频流
ACTIVE_SOURCE_TYPE = 'demo'

# 默认仍然使用 traffic.mp4
ACTIVE_VIDEO_SOURCE = TRAFFIC_VIDEO_PATH

# 网络视频流地址
NETWORK_STREAM_URL = None

# ==========================================================
# 两套功能各自独立的截图目录
# ==========================================================

# 普通车辆撞线抓拍
TRAFFIC_SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    'src',
    'static',
    'save_img'
)

# 车辆事故抓拍
ACCIDENT_SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    'src',
    'static',
    'save_accident'
)

# 如果文件夹不存在，就自动创建
os.makedirs(
    TRAFFIC_SAVE_DIR,
    exist_ok=True
)

os.makedirs(
    ACCIDENT_SAVE_DIR,
    exist_ok=True
)


# ==========================================================
# 普通交通视频源状态
# ==========================================================

# None = 尚未选择
# '0' = 使用摄像头
# 其他情况 = 使用 traffic.mp4
camera = None

# 原项目变量，暂时保留，防止其他代码还引用它
file_streams = None


# ==========================================================
# 四个交通统计区域
# ==========================================================

list_pts_left_1 = []
list_pts_up_1 = []
list_pts_right_1 = []
list_pts_down_1 = []


# ==========================================================
# 当前交通视频尺寸
# ==========================================================

w = 0
h = 0


@app.route('/')
def web_shou():
    return render_template('start_v.html')


@app.route('/start_v.html')
def start():
    return render_template('start_v.html')


@app.route('/Web-shou.html')
def web_shou_():
    return render_template('Web-shou.html')
@app.route('/network_source.html')
def network_source():
    return render_template(
        'network_source.html'
    )


@app.route(
    '/connect_network_stream',
    methods=['POST']
)
def connect_network_stream():
    global NETWORK_STREAM_URL
    global ACTIVE_SOURCE_TYPE
    global ACTIVE_VIDEO_SOURCE

    # =====================================
    # 1. 获取前端发送的视频流地址
    # =====================================

    data = request.get_json(
        silent=True
    )

    if not data:

        return {
            'success': False,
            'message': '没有收到视频流地址'
        }

    stream_url = str(
        data.get('url', '')
    ).strip()


    # =====================================
    # 2. 判断是否为空
    # =====================================

    if not stream_url:

        return {
            'success': False,
            'message': '视频流地址不能为空'
        }


    # =====================================
    # 3. 只接受网络视频地址
    # =====================================

    allowed_prefix = (
        'http://',
        'https://',
        'rtsp://'
    )

    if not stream_url.lower().startswith(
        allowed_prefix
    ):

        return {
            'success': False,
            'message':
                '目前仅支持 HTTP、HTTPS、RTSP 网络视频流'
        }


    print('----------------------------------')
    print('正在连接网络视频流：')
    print(stream_url)
    print('----------------------------------')


    # =====================================
    # 4. 尝试让 OpenCV 打开视频流
    # =====================================

    cap = cv2.VideoCapture(
        stream_url
    )


    if not cap.isOpened():

        cap.release()

        return {
            'success': False,
            'message':
                '无法打开该网络视频流，请检查地址是否正确或是否可公开访问'
        }


    # =====================================
    # 5. 尝试真正读取一帧
    # =====================================

    success, frame = cap.read()

    cap.release()


    if not success or frame is None:

        return {
            'success': False,
            'message':
                '已经连接视频源，但无法读取视频画面'
        }


    # =====================================
    # 6. 保存第一帧
    # =====================================

    photo_path = os.path.join(
        PROJECT_ROOT,
        'src',
        'static',
        'photo',
        '1.jpg'
    )

    save_success = cv2.imwrite(
        photo_path,
        frame
    )


    if not save_success:

        return {
            'success': False,
            'message':
                '网络视频读取成功，但首帧保存失败'
        }


    # =====================================
    # 7. 正式切换系统视频源
    # =====================================

    NETWORK_STREAM_URL = stream_url

    ACTIVE_SOURCE_TYPE = 'network'

    ACTIVE_VIDEO_SOURCE = stream_url


    print('----------------------------------')
    print('网络视频流连接成功')
    print('当前模式：实时道路监测')
    print(
        '视频尺寸：',
        frame.shape[1],
        'x',
        frame.shape[0]
    )
    print('----------------------------------')


    # =====================================
    # 8. 告诉前端连接成功
    # =====================================

    return {
        'success': True,
        'message': '网络道路视频流连接成功',
        'redirect': '/web.html'
    }


@app.route('/web_config.html')
def config():
    return render_template('web_config.html')


@app.route('/web.html', methods=['GET', 'POST'])
def web():
    return render_template('web.html')


@app.route('/photo_1.html')
def photo_1():
    return render_template('photo_1.html')


@app.route('/photo_2.html')
def photo_2():
    return render_template('photo_2.html')


@app.route('/web-analyse.html')
def web_analyse():
    return render_template('web-analyse.html')


@app.route('/web_get', methods=['GET'])
def web_get():
    text_1 = request.args.getlist('text')
    text = []

    # 读取当前演示视频第一帧，得到真实视频尺寸
    frame = cv2.imread('./src/static/photo/1.jpg')
    frame_h, frame_w = frame.shape[:2]

    # HTML canvas 内部实际坐标为 300 x 150
    for index, value in enumerate(text_1):
        if index % 2 == 0:
            # X：300 -> 视频真实宽度
            text.append(int(float(value) * frame_w / 300))
        else:
            # Y：150 -> 视频真实高度
            text.append(int(float(value) * frame_h / 150))
    global list_pts_left_1
    list_pts_left_1 = [[text[0], text[1]], [text[2], text[3]], [text[4], text[5]], [text[6], text[7]]]
    global list_pts_up_1
    list_pts_up_1 = [[text[8], text[9]], [text[10], text[11]], [text[12], text[13]], [text[14], text[15]]]
    global list_pts_right_1
    list_pts_right_1 = [[text[16], text[17]], [text[18], text[19]], [text[20], text[21]], [text[22], text[23]]]
    global list_pts_down_1
    list_pts_down_1 = [[text[24], text[25]], [text[26], text[27]], [text[28], text[29]], [text[30], text[31]]]
    print(text_1)
    print(text)
    return render_template('web_use.html')


@app.route('/web_get_file', methods=['GET'])
def web_get_file():

    global ACTIVE_SOURCE_TYPE
    global ACTIVE_VIDEO_SOURCE

    # =====================================
    # 当前进入的是“典型场景演示”
    # =====================================
    ACTIVE_SOURCE_TYPE = 'demo'
    ACTIVE_VIDEO_SOURCE = TRAFFIC_VIDEO_PATH

    print('----------------------------------')
    print('当前模式：典型场景演示')
    print('视频源：', ACTIVE_VIDEO_SOURCE)
    print('----------------------------------')

    cap = cv2.VideoCapture(
        ACTIVE_VIDEO_SOURCE
    )

    success, frame = cap.read()

    cap.release()

    if not success or frame is None:

        return (
            'traffic.mp4 无法读取，请检查文件',
            500
        )

    # 保存第一帧
    # 后面的区域选择页面继续使用这张图片
    photo_path = os.path.join(
        PROJECT_ROOT,
        'src',
        'static',
        'photo',
        '1.jpg'
    )

    cv2.imwrite(
        photo_path,
        frame
    )

    return render_template(
        'web.html'
    )


@app.route('/web_use1.html')
def web_use():
    return render_template('web_use1.html')


@app.route("/json_dict_style.json", methods=['GET'])
def get_json_1():
    return render_template('json_dict_style.json')


@app.route("/json_dict.json", methods=['GET'])
def get_json_2():
    return render_template('json_dict.json')


@app.route("/json_dict_sum.json", methods=['GET'])
def get_json_3():
    return render_template('json_dict_sum.json')


@app.route("/json_dict_label.json", methods=['GET'])
def get_json_5():
    return render_template('json_dict_label.json')


@app.route("/detector_config.json", methods=['GET'])
def get_json_4():
    return render_template('detector_config.json')


@app.route("/get_config", methods=['GET'])
def get_config():
    img_size = request.args.get('img_size')
    threshold = request.args.get('threshold')
    stride = request.args.get('stride')
    weight = request.args.get('weight')
    device = request.args.get('device')
    # todo
    t = {'msg': '修改成功'}
    return json.dumps(t, ensure_ascii=False)
# ==========================================================
# 网络实时视频最新帧读取器
#
# 原理：
# 后台线程持续读取网络视频；
# 永远只保存最新一帧；
# 如果YOLO处理不过来，旧帧会被自动覆盖，
# 防止实时视频产生越来越大的延迟。
# ==========================================================

class LatestNetworkFrame:

    def __init__(self, source):

        self.source = source

        self.cap = None

        self.frame = None

        self.frame_id = 0

        self.running = False

        self.lock = threading.Lock()

        self.thread = None


    def start(self):

        # 打开网络视频
        self.cap = cv2.VideoCapture(
            self.source
        )

        if not self.cap.isOpened():

            print(
                '[NETWORK] 无法打开网络视频流'
            )

            return False


        # 尽量缩小OpenCV内部缓冲
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        # 先读取第一帧
        ok, frame = self.cap.read()

        if not ok or frame is None:

            print(
                '[NETWORK] 无法读取网络视频第一帧'
            )

            self.cap.release()

            return False


        # 保存第一帧
        with self.lock:

            self.frame = frame

            self.frame_id = 1


        # 启动后台线程
        self.running = True

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )

        self.thread.start()

        print(
            '[NETWORK] 最新帧读取线程已启动'
        )

        return True


    def _update(self):

        while self.running:

            ok, frame = self.cap.read()

            if not ok or frame is None:

                # 网络偶尔丢一帧时不要马上退出
                time.sleep(0.02)

                continue


            # 直接覆盖旧帧
            with self.lock:

                self.frame = frame

                self.frame_id += 1


    def read(self):

        with self.lock:

            if self.frame is None:

                return False, None, self.frame_id


            return (
                True,
                self.frame.copy(),
                self.frame_id
            )


    def release(self):

        self.running = False

        if self.cap is not None:

            self.cap.release()

            self.cap = None
def detect_gen():
    global ACTIVE_VIDEO_SOURCE
    global ACTIVE_SOURCE_TYPE
    global h
    global w

    # ==============================
    # 当前交通分析所使用的视频源
    # ==============================

    video_source = ACTIVE_VIDEO_SOURCE

    print('----------------------------------')
    print('启动交通检测')
    print('视频源类型：', ACTIVE_SOURCE_TYPE)
    print('当前视频源：', video_source)
    print('----------------------------------')

    # ==================================================
    # 视频读取方式
    #
    # demo:
    #   traffic.mp4正常顺序读取
    #
    # network:
    #   后台持续取最新帧，避免直播积帧变成慢动作
    # ==================================================

    capture = None

    network_reader = None

    network_frame_id = -1

    if ACTIVE_SOURCE_TYPE == 'network':

        print(
            '[VIDEO] 使用网络实时最新帧模式'
        )

        network_reader = LatestNetworkFrame(
            video_source
        )

        if not network_reader.start():
            print(
                '[ERROR] 网络视频流启动失败:',
                video_source
            )

            return

        ok, img, network_frame_id = (
            network_reader.read()
        )


    else:

        print(
            '[VIDEO] 使用本地视频顺序读取模式'
        )

        capture = cv2.VideoCapture(
            video_source
        )

        if not capture.isOpened():
            print(
                '[ERROR] 无法打开交通视频:',
                video_source
            )

            return

        ok, img = capture.read()

    # 第一帧读取失败
    if not ok or img is None:

        print(
            '[ERROR] 无法读取交通视频第一帧:',
            video_source
        )

        if capture is not None:
            capture.release()

        if network_reader is not None:
            network_reader.release()

        return

    size_img = img.shape

    w = size_img[1]
    h = size_img[0]

    print('[VIDEO] 视频尺寸:', w, 'x', h)
    print('[VIDEO] 视频源连接成功')
    global list_pts_left_1
    global list_pts_up_1
    global list_pts_right_1
    global list_pts_down_1

    # 原视频大小, (宽高
    img_size_start_h = int(h)
    img_size_start_w = int(w)
    # 变换后进入处理视频的尺寸大小 (宽/2, 高/2
    img_size_w = int(img_size_start_w / 2)
    img_size_h = int(img_size_start_h / 2)

    # 定位框位置，判断下一步方向
    # TODO
    # list_pts_left = [[480, 583], [140, 790], [165, 850], [570, 600]]
    # list_pts_up = [[690, 530], [900, 550], [870, 610], [650, 590]]
    # list_pts_right = [[930, 950], [735, 1920], [830, 1920], [1000, 970]]
    # list_pts_down = [[10, 1200], [50, 1150], [630, 1920], [540, 1920]]
    list_pts_left = list_pts_left_1
    list_pts_up = list_pts_up_1
    list_pts_right = list_pts_right_1
    list_pts_down = list_pts_down_1

    # 初始化4个撞线polygon
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((img_size_start_h, img_size_start_w), dtype=np.uint8)
    ndarray_pts_left = np.array(list_pts_left, np.int32)
    polygon_value_left = cv2.fillPoly(mask_image_temp, [ndarray_pts_left], color=1)
    polygon_value_left = polygon_value_left[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((img_size_start_h, img_size_start_w), dtype=np.uint8)
    ndarray_pts_up = np.array(list_pts_up, np.int32)
    polygon_value_up = cv2.fillPoly(mask_image_temp, [ndarray_pts_up], color=2)
    polygon_value_up = polygon_value_up[:, :, np.newaxis]

    # 填充第三个polygon
    mask_image_temp = np.zeros((img_size_start_h, img_size_start_w), dtype=np.uint8)
    ndarray_pts_right = np.array(list_pts_right, np.int32)
    polygon_value_right = cv2.fillPoly(mask_image_temp, [ndarray_pts_right], color=3)
    polygon_value_right = polygon_value_right[:, :, np.newaxis]

    # 填充第四个polygon
    mask_image_temp = np.zeros((img_size_start_h, img_size_start_w), dtype=np.uint8)
    ndarray_pts_down = np.array(list_pts_down, np.int32)
    polygon_value_down = cv2.fillPoly(mask_image_temp, [ndarray_pts_down], color=4)
    polygon_value_down = polygon_value_down[:, :, np.newaxis]

    # 撞线检测用mask，包含4个polygon，（值范围 0 1 2 3 4），供撞线计算使用
    polygon_mask_left_up_right_down = polygon_value_left + polygon_value_up + polygon_value_right + polygon_value_down

    # 缩小尺寸
    polygon_mask_left_up_right_down = cv2.resize(polygon_mask_left_up_right_down, (img_size_w, img_size_h))

    # 左 色盘 与 polygon图片
    left_color_plate = [255, 0, 0]
    left_image = np.array(polygon_value_left * left_color_plate, np.uint8)

    # 上 色盘 与 polygon图片
    up_color_plate = [0, 255, 255]
    up_image = np.array(polygon_value_up * up_color_plate, np.uint8)

    # 右 色盘 与 polygon图片
    right_color_plate = [0, 255, 0]
    right_image = np.array(polygon_value_right * right_color_plate, np.uint8)

    # 下 色盘 与 polygon图片
    down_color_plate = [255, 0, 255]
    down_image = np.array(polygon_value_down * down_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = left_image + up_image + right_image + down_image

    # 缩小尺寸
    color_polygons_image = cv2.resize(color_polygons_image, (img_size_w, img_size_h))

    # list 与左侧polygon重叠
    list_overlapping_left_polygon = []

    # list 与上侧polygon重叠
    list_overlapping_up_polygon = []

    # list 与右侧polygon重叠
    list_overlapping_right_polygon = []

    # list 与下侧polygon重叠
    list_overlapping_down_polygon = []

    # 从左到右
    left2right_count = 0
    # 从左到上
    left2up_count = 0
    # 从左到下
    left2down_count = 0

    # 从上到左
    up2left_count = 0
    # 从上到右
    up2right_count = 0
    # 从上到下
    up2down_count = 0

    # 从右到左
    right2left_count = 0
    # 从右到上
    right2up_count = 0
    # 从右到下
    right2down_count = 0

    # 从下到左
    down2left_count = 0
    # 从下到上
    down2up_count = 0
    # 从下到右
    down2right_count = 0

    # ==========================================================
    # 初始化 YOLO26 + TrackTrack
    # ==========================================================

    traffic_tracker = TrafficTracker26()
    traffic_analyzer = TrafficAnalyzer()

    # TrafficAnalyzer 使用的视频帧计数
    analysis_frame_index = 0
    # 设备使用情况
    if torch.cuda.is_available():
        print(
            '当前使用的设备为 {}'.format(
                torch.cuda.get_device_name()
            )
        )
    else:
        print(
            '当前使用的设备为 {}'.format(
                'CPU'
            )
        )
    # json转换数据存储字典
    json_dict = dict()
    # json转换数据状态字典
    json_dict_style = dict()
    # json转换数据label与id字典
    json_dict_label = dict()
    # json转换数据经流总数字典
    json_dict_sum = dict()
    # 用于转换的信息数组,[] to dict()
    list_dic = []

    # ==============================
    # 最佳抓拍帧缓存
    # ==============================

    # 正在等待最佳抓拍的车辆
    pending_shots = {}

    # 已经完成抓拍的车辆
    captured_ids = set()

    # 撞线以后继续观察多少帧
    # 目前约 20 FPS，12帧大约0.6秒
    BEST_SHOT_FRAMES = 12
    #
    list_sum = []
    # ==========================================================
    # 前端统计数据写入频率
    #
    # AI仍然每帧检测
    # 只是JSON文件每0.2秒写一次
    # ==========================================================

    JSON_WRITE_INTERVAL = 0.2

    last_json_write_time = 0.0

    base_filename = os.path.join(
        os.getcwd(),
        'src'
    )
    def calc_shot_score(frame, c1, c2, scale_x, scale_y):
        """
        综合评价抓拍质量：
        1. 车辆越大越好
        2. 图像越清晰越好
        """

        if frame is None or frame.size == 0:
            return 0

        h, w = frame.shape[:2]

        # 检测坐标映射到原始高清帧
        x1 = max(0, int(c1[0] * scale_x))
        y1 = max(0, int(c1[1] * scale_y))
        x2 = min(w, int(c2[0] * scale_x))
        y2 = min(h, int(c2[1] * scale_y))

        if x2 <= x1 or y2 <= y1:
            return 0

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return 0

        # 太小的目标不参与最佳帧竞争
        roi_h, roi_w = roi.shape[:2]

        # 太远的小目标不作为最终抓拍
        if roi_w < 45 or roi_h < 35:
            return 0

        # -------------------------
        # 清晰度：拉普拉斯方差
        # 数值越大通常越清晰
        # -------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        # -------------------------
        # 车辆大小
        # -------------------------
        area = roi_w * roi_h

        # 综合评分
        # sqrt(area) 防止“大”完全压过“清晰”
        # 车辆尺寸优先，清晰度作为辅助
        score = area * np.log1p(sharpness)

        return score

    def queue_best_shot(track_id, label, c1, c2):
        """
        车辆撞线后进入最佳抓拍队列
        """

        if track_id in captured_ids:
            return

        if track_id in pending_shots:
            return

        crop = build_shot_crop(
            frame_raw,
            c1,
            c2,
            track_id,
            label,
            scale_x,
            scale_y
        )

        score = calc_shot_score(
            frame_raw,
            c1,
            c2,
            scale_x,
            scale_y
        )

        pending_shots[track_id] = {
            'frames_left': BEST_SHOT_FRAMES,
            'best_score': score,
            'best_crop': crop
        }

    while True:

        # ==================================================
        # 网络实时流：
        # 只获取后台线程保存的最新帧
        # ==================================================

        if ACTIVE_SOURCE_TYPE == 'network':

            ok, im, current_frame_id = (
                network_reader.read()
            )

            # 当前暂时没有可用帧
            if not ok or im is None:
                time.sleep(0.01)

                continue

            # 后台还没有产生新帧
            # 不要重复检测同一张图片
            if current_frame_id == network_frame_id:
                time.sleep(0.002)

                continue

            # 记录已经处理到哪一帧
            network_frame_id = current_frame_id


        # ==================================================
        # 本地演示视频：
        # 保持原来的顺序读取方式
        # ==================================================

        else:

            ok, im = capture.read()

            if not ok or im is None:
                print(
                    'The path or video/image is None'
                )

                break
        # 不再复制整张高清画面
        # 后面的cv2.resize会创建新图，因此这里直接引用即可
        frame_raw = im

        raw_h, raw_w = frame_raw.shape[:2]
        # 初始化person与vehicle总数量
        # 定义为局部变量,每一次循环进行一次更新
        sum_person = 0
        sum_vehicle = 0
        # 计算实时帧率开始时间
        start_time = time.time()

        # 缩小尺寸
        im = cv2.resize(im, (img_size_w, img_size_h))
        # 半分辨率检测框 -> 原始视频坐标的比例
        scale_x = raw_w / float(img_size_w)
        scale_y = raw_h / float(img_size_h)
        # ==========================================================
        # YOLO26 + TrackTrack
        # ==========================================================

        list_bboxs = traffic_tracker.track(im)
        # ==========================================================
        # TrafficAnalyzer
        # ==========================================================

        analysis_frame_index += 1

        if ACTIVE_SOURCE_TYPE == 'network':

            # 网络直播使用真实经过时间
            analysis_timestamp = time.monotonic()

        else:

            # 本地视频使用视频自身时间轴
            video_time_ms = capture.get(
                cv2.CAP_PROP_POS_MSEC
            )

            if video_time_ms >= 0:

                analysis_timestamp = (
                        video_time_ms / 1000.0
                )

            else:

                # 某些视频后端无法提供时间戳时
                # 使用视频 FPS 作为备用时间轴
                source_fps = capture.get(
                    cv2.CAP_PROP_FPS
                )

                if source_fps <= 0:
                    source_fps = 30.0

                analysis_timestamp = (
                        analysis_frame_index
                        /
                        source_fps
                )

        traffic_analysis = traffic_analyzer.update(
            tracks=list_bboxs,
            timestamp=analysis_timestamp,
            frame_shape=im.shape
        )

        # 每 60 帧打印一次分析结果
        # 不影响每帧 AI 推理
        if analysis_frame_index % 60 == 0:
            print(
                '[ANALYZER] '
                '车辆={} '
                '平均速度={:.2f}px/s '
                '移动={} '
                '静止={} '
                '静止比例={:.2%} '
                '占用率={:.2%} '
                '车型={}'.format(
                    traffic_analysis['vehicle_count'],
                    traffic_analysis['average_speed_px_s'],
                    traffic_analysis['moving_vehicle_count'],
                    traffic_analysis['stopped_vehicle_count'],
                    traffic_analysis['stopped_ratio'],
                    traffic_analysis['occupancy_ratio'],
                    traffic_analysis['class_counts']
                )
            )

        # ==========================================================
        # 绘制跟踪结果
        # ==========================================================

        output_image_frame = im.copy()

        for item_bbox in list_bboxs:
            x1, y1, x2, y2, label, track_id, confidence = item_bbox

            # 画检测框
            cv2.rectangle(
                output_image_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # 标签
            title = '{} ID:{} {:.2f}'.format(
                label,
                track_id,
                confidence
            )

            cv2.putText(
                output_image_frame,
                title,
                (
                    int(x1),
                    max(20, int(y1) - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        # =========================================
        # 对已经撞线的车辆继续观察，自动挑选最大的一帧
        # =========================================

        for item_bbox in list_bboxs:
            x1, y1, x2, y2, label, track_id, _ = item_bbox

            if track_id not in pending_shots:
                continue

            c1 = (x1, y1)
            c2 = (x2, y2)

            # 检测框越大，一般表示车辆距离摄像头越近
            score = calc_shot_score(
                frame_raw,
                c1,
                c2,
                scale_x,
                scale_y
            )

            if score > pending_shots[track_id]['best_score']:

                crop = build_shot_crop(
                    frame_raw,
                    c1,
                    c2,
                    track_id,
                    label,
                    scale_x,
                    scale_y
                )

                if crop is not None:
                    pending_shots[track_id]['best_score'] = score
                    pending_shots[track_id]['best_crop'] = crop

        # 所有正在等待抓拍的车辆倒计时
        for track_id in list(pending_shots.keys()):

            pending_shots[track_id]['frames_left'] -= 1

            if pending_shots[track_id]['frames_left'] <= 0:
                save_shot_crop(
                    pending_shots[track_id]['best_crop'],
                    track_id
                )

                captured_ids.add(track_id)

                del pending_shots[track_id]
        # 输出图片(带有标注线)
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # 实时交通状态显示
            if len(list_bboxs) > 20:
                style = '拥堵'
            else:
                style = '正常'
            json_dict['style'] = style
            json_dict['len_list'] = str(len(list_bboxs))
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id, _ = item_bbox

                # 保存点（对角点）
                c1, c2 = (x1, y1), (x2, y2)
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0， x方向偏移量 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                x1_offset = int(x1 + ((x2 - x1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1_offset
                # 统计实时流量
                for key in json_dict_label:
                    if key != track_id:
                        json_dict_label[track_id] = label
                list_sum.append(label)
                # 将流量总数存入json
                for label_sum in list_sum:
                    if label_sum in ['car', 'bus', 'truck']:
                        sum_vehicle += 1
                    if label_sum == 'person':
                        sum_person += 1
                list_sum.clear()
                json_dict_sum['Sum_person'] = sum_person
                json_dict_sum['Sum_vehicle'] = sum_vehicle
                # 如果撞 左polygon(左上右下)
                if polygon_mask_left_up_right_down[y, x] == 1:
                    if track_id not in list_overlapping_left_polygon:
                        list_overlapping_left_polygon.append(track_id)
                    pass

                    # 判断 上polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 由上到左方向
                    if track_id in list_overlapping_up_polygon:
                        # 上到左+1
                        up2left_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 上polygon list 中的此id
                        list_overlapping_up_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass
                    # 判断 右polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 由右到左方向
                    elif track_id in list_overlapping_right_polygon:
                        # 右到左+1
                        right2left_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 右polygon list 中的此id
                        list_overlapping_right_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass
                    # 判断 下polygon list 是否有此 track_id
                    # 有此 track_id, 则认为是 由下到左方向
                    elif track_id in list_overlapping_down_polygon:
                        # 下到左+1
                        down2left_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 下polygon list 中的此id
                        list_overlapping_down_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass
                    else:
                        # 没有此id
                        pass

                elif polygon_mask_left_up_right_down[y, x] == 2:
                    # 如果撞 上polygon
                    if track_id not in list_overlapping_up_polygon:
                        list_overlapping_up_polygon.append(track_id)
                    pass

                    # 判断 左polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 左到上方向
                    if track_id in list_overlapping_left_polygon:
                        # 左到上+1
                        left2up_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 左polygon list 中的此id
                        list_overlapping_left_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )

                        pass
                    elif track_id in list_overlapping_right_polygon:
                        # 右到上+1
                        right2up_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 右polygon list 中的此id
                        list_overlapping_right_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass

                    elif track_id in list_overlapping_down_polygon:
                        # 下到上+1
                        down2up_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 下polygon list 中的此id
                        list_overlapping_down_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass

                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                elif polygon_mask_left_up_right_down[y, x] == 3:
                    # 如果撞 右polygon
                    if track_id not in list_overlapping_right_polygon:
                        list_overlapping_right_polygon.append(track_id)
                    pass

                    # 判断 左polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 左到右方向
                    if track_id in list_overlapping_left_polygon:
                        # 左到右+1
                        left2right_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 左polygon list 中的此id
                        list_overlapping_left_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )

                        pass
                    # 如果在上中
                    elif track_id in list_overlapping_up_polygon:
                        # 上到右+1
                        up2right_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除上polygon中此id
                        list_overlapping_up_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass
                    # 如果在下中
                    elif track_id in list_overlapping_down_polygon:
                        # 下到右+=1
                        down2right_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除下polygon中此id
                        list_overlapping_down_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                elif polygon_mask_left_up_right_down[y, x] == 4:
                    # 如果撞 下polygon
                    if track_id not in list_overlapping_down_polygon:
                        list_overlapping_down_polygon.append(track_id)
                    pass

                    # 判断 左polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 左到下方向
                    if track_id in list_overlapping_left_polygon:
                        # 左到右+1
                        left2down_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 左polygon list 中的此id
                        list_overlapping_left_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )

                        pass
                    # 上
                    elif track_id in list_overlapping_up_polygon:
                        # 上到下+=1
                        up2down_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除上中此id
                        list_overlapping_up_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass

                    # 右
                    elif track_id in list_overlapping_right_polygon:
                        # 右到下+=1
                        right2down_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除右中此 id
                        list_overlapping_right_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        queue_best_shot(
                            track_id,
                            label,
                            c1,
                            c2
                        )
                        pass

                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_left_polygon + list_overlapping_up_polygon \
                                   + list_overlapping_right_polygon + list_overlapping_down_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id, _ in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_left_polygon:
                        list_overlapping_left_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_up_polygon:
                        list_overlapping_up_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_right_polygon:
                        list_overlapping_right_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_down_polygon:
                        list_overlapping_down_polygon.remove(id1)
                    pass
                pass
            # 清空list
            list_overlapping_all.clear()
            pass

            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_left_polygon.clear()
            list_overlapping_up_polygon.clear()
            list_overlapping_right_polygon.clear()
            list_overlapping_down_polygon.clear()
            pass
        pass

        json_dict_style.setdefault('UP_LEFT', str(up2left_count))
        json_dict_style.setdefault('UP_RIGHT', str(up2right_count))
        json_dict_style.setdefault('UP_DOWN', str(up2down_count))
        json_dict_style.setdefault('LEFT_RIGHT', str(left2right_count))
        json_dict_style.setdefault('LEFT_UP', str(left2up_count))
        json_dict_style.setdefault('LEFT_DOWN', str(left2down_count))
        json_dict_style.setdefault('RIGHT_UP', str(right2up_count))
        json_dict_style.setdefault('RIGHT_LEFT', str(right2left_count))
        json_dict_style.setdefault('RIGHT_DOWN', str(right2down_count))
        json_dict_style.setdefault('DOWN_LEFT', str(down2left_count))
        json_dict_style.setdefault('DOWN_UP', str(down2up_count))
        json_dict_style.setdefault('DOWN_RIGHT', str(down2right_count))

        # 计算实时帧率结束时间
        end_time = time.time()
        # 计算fps并打印
        seconds = end_time - start_time
        fps = 1 / seconds
        fps = '%.2f' % fps
        json_dict_style.setdefault('FPS', fps)

        # # 判断label文件内容是否唯一
        # length = len(list_dic)
        # for i in range(length):
        #     for j in range(i):
        #         if list_dic[i] == list_dic[j]:
        #             list_dic.remove(list_dic[i])
        # print(list_dic)
        # ==========================================================
        # 降低磁盘JSON写入频率
        #
        # YOLO/DeepSORT仍然每帧工作
        # 前端统计数据每0.2秒更新一次
        # ==========================================================

        now_write_time = time.time()

        if (
                now_write_time
                - last_json_write_time
                >= JSON_WRITE_INTERVAL
        ):

            try:

                if json_dict:
                    with open(
                            os.path.join(
                                base_filename,
                                'templates',
                                'json_dict.json'
                            ),
                            'w+'
                    ) as f_1:
                        json.dump(
                            json_dict,
                            f_1
                        )

                with open(
                        os.path.join(
                            base_filename,
                            'templates',
                            'json_dict_style.json'
                        ),
                        'w+'
                ) as f_2:

                    json.dump(
                        json_dict_style,
                        f_2
                    )

                with open(
                        os.path.join(
                            base_filename,
                            'templates',
                            'json_dict_label.json'
                        ),
                        'w+'
                ) as f_3:

                    json.dump(
                        list_dic,
                        f_3
                    )

                with open(
                        os.path.join(
                            base_filename,
                            'templates',
                            'json_dict_sum.json'
                        ),
                        'w+'
                ) as f_4:

                    json.dump(
                        json_dict_sum,
                        f_4
                    )

                last_json_write_time = (
                    now_write_time
                )


            except IOError:

                print(
                    "文件写入失败"
                )

        output_image_frame = cv2.resize(output_image_frame, (img_size_start_w, img_size_start_h))
        frame = output_image_frame
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

        # 清空方式以及总数相关信息的字典, 方便后面写入以及json的数据的读取
        json_dict.clear()
        json_dict_style.clear()


# ==========================================================
# 独立事故检测模块
# ==========================================================

def save_accident_snapshot(frame, bbox, shot_id):
    """
    保存一次事故抓拍。
    bbox:
    (x1, y1, x2, y2, label, confidence)
    """

    if frame is None or frame.size == 0:
        return False

    x1, y1, x2, y2, label, conf = bbox

    h, w = frame.shape[:2]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # 留20%背景，方便看清事故环境
    pad_x = max(20, int(box_w * 0.20))
    pad_y = max(20, int(box_h * 0.20))

    sx1 = max(0, x1 - pad_x)
    sy1 = max(0, y1 - pad_y)

    sx2 = min(w, x2 + pad_x)
    sy2 = min(h, y2 + pad_y)

    if sx2 <= sx1 or sy2 <= sy1:
        return False

    crop = frame[sy1:sy2, sx1:sx2].copy()

    if crop.size == 0:
        return False

    confidence = float(conf)

    title = 'ACCIDENT {:.2f}'.format(confidence)

    cv2.rectangle(
        crop,
        (0, 0),
        (min(crop.shape[1], 250), 35),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        crop,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    path = os.path.join(
        ACCIDENT_SAVE_DIR,
        '{}.jpg'.format(shot_id)
    )

    return cv2.imwrite(
        path,
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )


def accident_gen():
    """
    独立事故检测视频流：

    accident.mp4
        ↓
    Detector_sg / best.pt
        ↓
    事故检测框
        ↓
    事故抓拍
    """

    # --------------------------------
    # accident.mp4不存在时显示提示画面
    # --------------------------------

    if not os.path.exists(ACCIDENT_VIDEO_PATH):

        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8
        )

        cv2.putText(
            frame,
            'ACCIDENT VIDEO NOT FOUND',
            (300, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            'Please put accident.mp4 in project root',
            (280, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        while True:

            ok, buffer = cv2.imencode(
                '.jpg',
                frame
            )

            if ok:
                yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + buffer.tobytes()
                        + b'\r\n'
                )

            time.sleep(1)

    # --------------------------------
    # 初始化事故模型
    # --------------------------------

    detector_sg = Detector_sg()

    cap = cv2.VideoCapture(
        ACCIDENT_VIDEO_PATH
    )

    if not cap.isOpened():
        print(
            '[ERROR] accident.mp4 无法打开:',
            ACCIDENT_VIDEO_PATH
        )

        return

    # 已有事故图片编号
    existing_ids = []

    for filename in os.listdir(
            ACCIDENT_SAVE_DIR
    ):

        name, ext = os.path.splitext(
            filename
        )

        if (
                ext.lower() == '.jpg'
                and name.isdigit()
        ):
            existing_ids.append(
                int(name)
            )

    shot_id = max(
        existing_ids,
        default=0
    ) + 1

    # --------------------------------
    # 同一事故只抓拍一次
    # 连续若干帧未检测到事故后
    # 才认为下一次是新事故
    # --------------------------------

    accident_active = False

    clear_frames = 0

    CLEAR_THRESHOLD = 15

    try:

        while True:

            ok, frame = cap.read()

            # 视频结束后循环播放
            if not ok or frame is None:
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue

            output = frame.copy()

            # ==========================
            # 真正的事故检测
            # ==========================

            bboxes_sg = detector_sg.detect(
                frame
            )

            # ==========================
            # 绘制所有事故框
            # ==========================

            for bbox in bboxes_sg:
                x1, y1, x2, y2, label, conf = bbox

                confidence = float(conf)

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                title = (
                    'ACCIDENT {:.2f}'
                    .format(confidence)
                )

                cv2.putText(
                    output,
                    title,
                    (
                        x1,
                        max(30, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # ==========================
            # 事故事件开始
            # ==========================

            if len(bboxes_sg) > 0:

                clear_frames = 0

                # 新事故第一次出现才保存
                if not accident_active:

                    accident_active = True

                    # 选择置信度最高事故框
                    best_box = max(
                        bboxes_sg,
                        key=lambda item: float(
                            item[5]
                        )
                    )

                    success = (
                        save_accident_snapshot(
                            frame,
                            best_box,
                            shot_id
                        )
                    )

                    if success:
                        print(
                            '[ACCIDENT] saved:',
                            shot_id
                        )

                        shot_id += 1

            else:

                # 当前帧没有事故
                if accident_active:

                    clear_frames += 1

                    # 连续15帧没事故
                    # 认为这次事故结束
                    if (
                            clear_frames
                            >= CLEAR_THRESHOLD
                    ):
                        accident_active = False

                        clear_frames = 0

            # ==========================
            # Flask MJPEG输出
            # ==========================

            ok, buffer = cv2.imencode(
                '.jpg',
                output
            )

            if not ok:
                continue

            yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + buffer.tobytes()
                    + b'\r\n'
            )

    finally:

        cap.release()


# ==========================================================
# 事故视频流路由
# ==========================================================

@app.route('/accident_video_feed')
def accident_video_feed():
    return Response(
        accident_gen(),
        mimetype=(
            'multipart/x-mixed-replace; '
            'boundary=frame'
        )
    )


# ==========================================================
# 返回当前所有事故抓拍图片
# ==========================================================

@app.route('/accident_images')
def accident_images():
    files = []

    if os.path.exists(
            ACCIDENT_SAVE_DIR
    ):

        for filename in os.listdir(
                ACCIDENT_SAVE_DIR
        ):

            name, ext = os.path.splitext(
                filename
            )

            if (
                    ext.lower() == '.jpg'
                    and name.isdigit()
            ):
                files.append(
                    filename
                )

    files.sort(
        key=lambda item: int(
            os.path.splitext(item)[0]
        )
    )

    return Response(
        json.dumps(
            files,
            ensure_ascii=False
        ),
        mimetype='application/json'
    )


@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == 'Camera_0':
        return Response(detect_gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


def Run():
    app.run(host='0.0.0.0', port="5000", threaded=True, debug=True)
