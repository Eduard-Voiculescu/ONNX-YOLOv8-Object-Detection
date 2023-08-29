from yolov8 import YOLOv8

import constant
import inotify.adapters
import watcher


def main():
    yolov8_detector = YOLOv8(constant.ONNX_MODEL_PATH, conf_thres=0.2, iou_thres=0.3)
    w = watcher.Watcher(yolov8_detector)
    w.add_watch(constant.UNPROCESSED_FRAMEKM)

    try:
        w.run()
    except inotify.adapters.TerminalEventException as tee:
        print("Bye")
    except KeyboardInterrupt as ki:
        print("Bye")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
