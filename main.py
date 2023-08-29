from yolov8 import YOLOv8

import constant
import watcher


def main():
    yolov8_detector = YOLOv8(constant.ONNX_MODEL_PATH, conf_thres=0.2, iou_thres=0.3)
    w = watcher.Watcher(yolov8_detector)
    w.add_watch(constant.UNPROCESSED_FRAMEKM)
    w.run()


if __name__ == "__main__":
    main()
