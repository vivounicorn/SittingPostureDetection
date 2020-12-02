# coding:utf-8

import argparse
import json
import logging
import sys
import time

import PIL
import torch

import cv2
from src.posture import Posture
from src.config import Config
from src.logger import Logger
import openpifpaf
from openpifpaf import decoder, network, show, transforms, visualizer, __version__
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

logger = Logger(__name__).getLogger()


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__
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
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
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


class CustomFormatter:
    def __init__(self,
                 cfg_path='/home/zhanglei/Gitlab/SittingPostureDetection/config/cfg.ini'):

        self.cfg = Config(cfg_path)
        self.args = cli()
        self.processor, self.model = self.processor_factory()

        self.is_right = False

        self.keypoint_painter = show.KeypointPainter(color_connections=True, linewidth=6)
        self.annotation_painter = show.AnnotationPainter(keypoint_painter=self.keypoint_painter)

    def processor_factory(self):
        model, _ = network.factory_from_args(self.args)
        model = model.to(self.args.device)
        processor = decoder.factory_from_args(self.args, model)
        return processor, model

    def camera_calibration(self, fdirect=lambda x: "right" if x else "left"):
        cap = cv2.VideoCapture(self.args.source)
        logger.info("camera calibration start: please press key 'q' to capture snapshot.")

        while True:
            ret, frame = cap.read()
            cv2.putText(frame,
                        "The calibration direction is to the %s." % (fdirect(self.is_right)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.imshow("Camera Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.is_right = True
                logger.debug("The calibration direction is to the right.")
            if key == ord('l'):
                self.is_right = False
                logger.info("The calibration direction is to the left.")
            elif key == ord('s'):
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
                self.cfg.flush()

                with openpifpaf.show.image_canvas(frame,
                                                  fig_file=self.cfg.camera_calibration_path() + "/calibration_an.jpg",
                                                  show=False) as ax:
                    self.keypoint_painter.annotations(ax, prediction)

                cv2.imwrite(self.cfg.camera_calibration_path() + "/calibration.jpg", frame)
                with open(self.cfg.camera_calibration_path() + "calibration.jpg.prediction.json", 'w') as f:
                    f.write(js)
                    f.write('\n')
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def inference(self):
        last_loop = time.time()
        capture = cv2.VideoCapture(self.args.source)

        animation = show.AnimationFrame(
            fig_init_args={'figsize': (5, 5)},
            show=self.args.show,
            second_visual=self.args.debug or self.args.debug_indices,
        )

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

            if frame_i % self.cfg.skip_frame() == 0:
                pass
                prediction = self.processor.batch(self.model,
                                                  torch.unsqueeze(processed_image, 0),
                                                  device=self.args.device)[0]

                self.annotation_painter.annotations(ax, prediction)

                p = Thread(target=self.multi_posture, args=(prediction,))
                p.setDaemon(True)
                p.start()

            cv2.putText(image,
                        'frame %d, loop time = %.3fs, FPS = %.3f' % (frame_i,
                                                                     time.time() - last_loop,
                                                                     1.0 / (time.time() - last_loop)),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            last_loop = time.time()

            plt.rcParams['keymap.quit'] = ''
            # axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
            bcut = Button(ax, '')
            bcut.on_clicked(plt_quit)

            ax.set_title('Posture Detected', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax.imshow(image)

    def multi_posture(self, prediction):
        js = json.dumps(eval(str([ann.json_data() for ann in prediction])))
        posture = Posture(self.cfg, js)
        posture.detect(self.is_right)
