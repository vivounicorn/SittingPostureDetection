# coding:utf-8

import argparse
import json
import logging
import os
import time

import PIL
import torch
import random

from PIL import Image
from torch.backends import cudnn

from src.detection.posture import Posture
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier
from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.color_tools import random_rgb2hex
from src.detection.bounding_box import BoundingBox
from src.tracking.tracker import *
import openpifpaf
from openpifpaf import decoder, network, show, transforms, visualizer, __version__
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
# matplotlib.use('TkAgg')
plt.switch_backend('TkAgg')

logger = Logger(__name__).getLogger()


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--source', default='0',
                        help='OpenCV source url. Integer for webcams. Supports rtmp streams.')
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('-c', '--calibration', default=False, action='store_true')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    parser.add_argument('--image', default='',
                        help='input images')
    group.add_argument('-r', '--right', default=False, action='store_true',
                       help='the calibration direction is right(default is left)')

    args = parser.parse_args()

    args.debug_images = False

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG

    logger.setLevel(log_level)

    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if len(args.source) == 1:
        args.source = int(args.source)

    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    logger.debug('neural network device: %s', args.device)

    return args


def plt_quit(event):
    plt.close('all')
    sys.exit()


def draw_box(ax, x, y, w, h, color, linewidth=1):
    if w < 5.0:
        x -= 2.0
        w += 4.0
    if h < 5.0:
        y -= 2.0
        h += 4.0
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x, y), w, h, fill=False, color=color, linewidth=linewidth))


def crop_img(img, path, bbox: BoundingBox):
    cropped = img.crop(bbox.get_crop_tuple())
    cropped.save(path)


def img_resize(img_r, imgsz=640):
    img_r = letterbox(img_r, new_shape=imgsz)[0]

    # Convert
    img_r = img_r[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_r = np.ascontiguousarray(img_r)
    return img_r


def yolo_detect(image0, model, modelc, device, half, imgsz=640):

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    image = img_resize(image0)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img = torch.from_numpy(image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred)

    # Apply Classifier
    pred = apply_classifier(pred, modelc, img, image0)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'

                plot_one_box(xyxy, image0, label=label, color=colors[int(cls)], line_thickness=1)
                return int(xyxy[0]), int(xyxy[1]), int(xyxy[2])-int(xyxy[0]), int(xyxy[3])-int(xyxy[1])

    return None


def draw_tracking(ax, bbox, image, tracker):
    bbox = tracker.update(image)
    draw_box(ax, bbox[0], bbox[1], bbox[2], bbox[3], random_rgb2hex())


class CustomFormatter:
    def __init__(self):

        cfg_path = os.getcwd() + '/config/cfg.ini'
        if not os.path.exists(cfg_path):
            raise IOError("The configuration file was not found.")

        self.cfg = Config(cfg_path)
        self.args = cli()
        self.processor, self.model = self.processor_factory()

        self.is_right = False

        self.keypoint_painter = show.KeypointPainter(color_connections=True, linewidth=6)
        self.annotation_painter = show.AnnotationPainter(keypoint_painter=self.keypoint_painter)

    def processor_factory(self):
        model, _ = network.factory_from_args(self.args)
        model = model.to(self.args.device)
        if not self.args.disable_cuda and torch.cuda.device_count() > 1:
            logger.info('Using multiple GPUs: %d', torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            model.base_net = model.base_net
            model.head_nets = model.head_nets
        processor = decoder.factory_from_args(self.args, model)
        return processor, model

    def camera_calibration(self, fdirect=lambda x: "right" if x else "left"):
        cap = cv2.VideoCapture(self.args.source)
        logger.info("camera calibration start: please press key 'q' to capture snapshot.")

        model, modelc, device, half, imgsz = self.yolo_load(self.cfg.yolov5s_path())

        while True:
            ret, frame = cap.read()
            cv2.putText(frame,
                        "The calibration direction is to the %s." % (fdirect(self.is_right)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            # to detect each frame.
            # bb = yolo_detect(frame, model, modelc, device, half)

            cv2.imshow("Camera Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.is_right = True
                logger.debug("The calibration direction is to the right.")
            if key == ord('l'):
                self.is_right = False
                logger.info("The calibration direction is to the left.")
            elif key == ord('s'):
                bb = yolo_detect(frame, model, modelc, device, half)
                bbox = BoundingBox(bb)
                self.cfg.set_bbox(str(bbox))
                self.cfg.flush()
                pil_frame = PIL.Image.fromarray(frame)
                crop_img(img=pil_frame, path=self.cfg.camera_calibration_path() + "/calibration_bbox.jpg", bbox=bbox)

                self.calibration(frame)
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def yolo_load(self, weights, imgsz=640):
        device = select_device(str(self.args.device))
        half = device.type != 'cpu'

        # Load model
        sys.path.insert(0, './yolov5')
        model = attempt_load(weights, map_location=device)
        imgsz = check_img_size(imgsz, s=model.stride.max())
        if half:
            model.half()  # to FP16

        modelc = load_classifier(name='resnet18', n=2)

        return model, modelc, device, half, imgsz

    def calibration(self, frame):
        image_pil = PIL.Image.fromarray(frame)
        processed_image, _, __ = transforms.EVAL_TRANSFORM(image_pil, [], None)
        CustomFormatter()
        prediction = self.processor.batch(self.model,
                                          torch.unsqueeze(processed_image, 0),
                                          device=self.args.device)[0]
        js = json.dumps(eval(str([ann.json_data() for ann in prediction])))
        posture = Posture(self.cfg, js)
        angle1, angle2 = posture.detect_angle(self.is_right)
        self.cfg.set_shoulder_waist_knee_angle(str(angle1))
        self.cfg.set_ear_shoulder_waist_angle(str(angle2))

        cv2.putText(frame,
                    "waist angle:%.2f, neck angle:%.2f." % (angle1, angle2),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        with openpifpaf.show.image_canvas(frame,
                                          fig_file=self.cfg.camera_calibration_path() + "/calibration_an.jpg",
                                          show=False) as ax:
            self.keypoint_painter.annotations(ax, prediction)
        cv2.imwrite(self.cfg.camera_calibration_path() + "/calibration.jpg", frame)
        with open(self.cfg.camera_calibration_path() + "calibration.jpg.prediction.json", 'w') as f:
            f.write(js)
            f.write('\n')

    def image_calibration(self):
        self.is_right = self.args.right
        img = cv2.imread(self.args.image)
        if img is None:
            return

        self.calibration(img)

    def inference(self):
        last_loop = time.time()
        capture = cv2.VideoCapture(self.args.source)
        bbox = self.cfg.bbox()

        animation = show.AnimationFrame(
            fig_init_args={'figsize': (5, 5)},
            show=self.args.show,
            second_visual=self.args.debug or self.args.debug_indices,
        )

        tracker = KCFTracker(True, True, True)

        for frame_i, (ax, ax_second) in enumerate(animation.iter()):

            _, image = capture.read()

            if image is None:
                logger.info('no more images captured')
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if ax is None:
                ax, ax_second = animation.frame_init(image)

            visualizer.BaseVisualizer.image(image)
            visualizer.BaseVisualizer.common_ax = ax_second

            start = time.time()
            image_pil = PIL.Image.fromarray(image)
            processed_image, _, __ = transforms.EVAL_TRANSFORM(image_pil, [], None)
            logger.debug('preprocessing time %.3fs', time.time() - start)

            if frame_i == 0:
                tracker.init(bbox, image)

            p = Thread(target=draw_tracking, args=(ax, bbox, image, tracker,))
            p.setDaemon(True)
            p.start()
            # p.join()

            if frame_i % self.cfg.skip_frame() == 0:
                prediction = self.processor.batch(self.model,
                                                  torch.unsqueeze(processed_image, 0),
                                                  device=self.args.device)[0]

                self.annotation_painter.annotations(ax, prediction)

                p = Thread(target=self.multi_posture, args=(prediction,))
                p.setDaemon(True)
                p.start()
                # p.join()

            cv2.putText(image,
                        'frame %d, loop time = %.3fs, FPS = %.3f' % (frame_i,
                                                                     time.time() - last_loop,
                                                                     1.0 / (time.time() - last_loop)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            last_loop = time.time()

            plt.rcParams['keymap.quit'] = ''
            bcut = Button(ax, '')
            bcut.on_clicked(plt_quit)

            ax.set_title('Posture Detected', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax.imshow(image)

    def multi_posture(self, prediction):
        js = json.dumps(eval(str([ann.json_data() for ann in prediction])))
        posture = Posture(self.cfg, js)
        posture.detect(self.is_right)

    def run(self):
        if self.args.calibration:
            if not self.args.image:
                self.camera_calibration()
            else:
                self.image_calibration()

        self.inference()
