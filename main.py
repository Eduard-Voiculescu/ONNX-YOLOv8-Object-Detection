from argparse import ArgumentParser
from yolov8 import YOLOv8

import constant
import inotify.adapters
import logging
import sys
import watcher


def main(args):
    arg_parser = ArgumentParser(prog=__file__, add_help=False)
    arg_parser.add_argument('-l', '--log-level', default='INFO',
                                  help='set log level')
    args, _ = arg_parser.parse_known_args(args)

    try:
        logging.basicConfig(format='[%(process)d] %(levelname)s: %(message)s', level=args.log_level)
    except ValueError:
        logging.error("Invalid log level: {}".format(args.log_level))
        sys.exit(1)
    
    logger = logging.getLogger(__name__)
    logger.info("Log level set: {}"
                .format(logging.getLevelName(logger.getEffectiveLevel())))

    yolov8_detector = YOLOv8(constant.ONNX_MODEL_PATH, conf_thres=0.2, iou_thres=0.3)
    logging.info('model {constant.ONNX_MODEL_PATH} initialized...')
    w = watcher.Watcher(yolov8_detector, logger)
    w.add_watch(constant.UNPROCESSED_FRAMEKM)

    try:
        logging.info('starting watcher...')
        w.run()
    except inotify.adapters.TerminalEventException as tee:
        logging.info('bye')
    except KeyboardInterrupt as ki:
        logging.info('bye')
    except Exception as e:
        raise e


if __name__ == "__main__":
    main(sys.argv[1:])
