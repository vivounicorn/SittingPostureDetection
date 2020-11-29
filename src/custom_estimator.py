"""Video demo application.
Use --scale=0.2 to reduce the input image size to 20%.
Use --json-output for headless processing.
Example commands:
    python3 -m pifpaf.video --source=0  # default webcam
    python3 -m pifpaf.video --source=1  # another webcam
    # streaming source
    python3 -m pifpaf.video --source=http://127.0.0.1:8080/video
    # file system source (any valid OpenCV source)
    python3 -m pifpaf.video --source=docs/coco/000000081988.jpg
Trouble shooting:
* MacOSX: try to prefix the command with "MPLBACKEND=MACOSX".
"""

import argparse
import json
import logging
import os
import time

import PIL
import torch

import cv2  # pylint: disable=import-error
from src.posture import Posture
from openpifpaf import decoder, network, show, transforms, visualizer, __version__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CustomFormatter:
    def __init__(self):
        self.args = self.cli()
        self.processor, self.model = self.processor_factory(self.args)

        # create keypoint painter
        self.keypoint_painter = show.KeypointPainter(color_connections=self.args.colored_connections, linewidth=6)
        self.annotation_painter = show.AnnotationPainter(keypoint_painter=self.keypoint_painter)

    def cli(self):  # pylint: disable=too-many-statements,too-many-branches
        parser = argparse.ArgumentParser(
            prog='python3 -m openpifpaf.video',
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
        parser.add_argument('--video-fps', default=show.AnimationFrame.video_fps, type=float)
        parser.add_argument('--show', default=False, action='store_true')
        parser.add_argument('--horizontal-flip', default=False, action='store_true')
        parser.add_argument('--no-colored-connections',
                            dest='colored_connections', default=True, action='store_false',
                            help='do not use colored connections to draw poses')
        parser.add_argument('--disable-cuda', action='store_true',
                            help='disable CUDA')
        parser.add_argument('--scale', default=1.0, type=float,
                            help='input image scale factor')
        parser.add_argument('--start-frame', type=int, default=0)
        parser.add_argument('--skip-frames', type=int, default=1)
        parser.add_argument('--max-frames', type=int)
        group = parser.add_argument_group('logging')
        group.add_argument('-q', '--quiet', default=False, action='store_true',
                           help='only show warning messages or above')
        group.add_argument('--debug', default=False, action='store_true',
                           help='print debug messages')
        args = parser.parse_args()

        args.debug_images = False

        # configure logging
        log_level = logging.INFO
        if args.quiet:
            log_level = logging.WARNING
        if args.debug:
            log_level = logging.DEBUG
        logging.basicConfig()
        logging.getLogger('openpifpaf').setLevel(log_level)
        logger.setLevel(log_level)

        network.configure(args)
        show.configure(args)
        visualizer.configure(args)
        show.AnimationFrame.video_fps = args.video_fps

        # check whether source should be an int
        if len(args.source) == 1:
            args.source = int(args.source)

        # add args.device
        args.device = torch.device('cpu')
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
        logger.debug('neural network device: %s', args.device)

        return args

    def processor_factory(self, args):
        model, _ = network.factory_from_args(args)
        model = model.to(args.device)
        processor = decoder.factory_from_args(args, model)
        return processor, model

    def inference(self):
        last_loop = time.time()
        capture = cv2.VideoCapture(self.args.source)

        animation = show.AnimationFrame(
            show=self.args.show,
            second_visual=self.args.debug or self.args.debug_indices,
        )
        for frame_i, (ax, ax_second) in enumerate(animation.iter()):
            _, image = capture.read()
            if image is None:
                logger.info('no more images captured')
                break

            if frame_i < self.args.start_frame:
                animation.skip_frame()
                continue

            if frame_i % self.args.skip_frames != 0:
                animation.skip_frame()
                continue

            if self.args.scale != 1.0:
                image = cv2.resize(image, None, fx=self.args.scale, fy=self.args.scale)
                logger.debug('resized image size: %s', image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.args.horizontal_flip:
                image = image[:, ::-1]

            if ax is None:
                ax, ax_second = animation.frame_init(image)
            visualizer.BaseVisualizer.image(image)
            visualizer.BaseVisualizer.common_ax = ax_second

            start = time.time()
            image_pil = PIL.Image.fromarray(image)
            processed_image, _, __ = transforms.EVAL_TRANSFORM(image_pil, [], None)
            logger.debug('preprocessing time %.3fs', time.time() - start)

            preds = self.processor.batch(self.model, torch.unsqueeze(processed_image, 0), device=self.args.device)[0]

            js = json.dumps(eval(str([ann.json_data() for ann in preds])))
            posture = Posture(js)
            posture.detect()

            ax.imshow(image)
            self.annotation_painter.annotations(ax, preds)

            logger.info('frame %d, loop time = %.3fs, FPS = %.3f',
                        frame_i,
                        time.time() - last_loop,
                        1.0 / (time.time() - last_loop))
            last_loop = time.time()

            if self.args.max_frames and frame_i >= self.args.start_frame + self.args.max_frames:
                break
