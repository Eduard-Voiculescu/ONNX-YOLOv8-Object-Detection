from yolov8 import YOLOv8
from PIL import Image

import cv2
import os

onnx_new_model_path = "devel/models/pvc.onnx"
images_folder = "devel/unprocessed_framekm/km_20230527_205605_10_0"
yolov8_detector = YOLOv8(onnx_new_model_path, conf_thres=0.2, iou_thres=0.3)

images = []
for f in os.listdir(images_folder):
    p = os.path.join(images_folder, f)
    img = cv2.imread(p, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

for i, img in enumerate(images):
    boxes, scores, classe_ids = yolov8_detector(img)
    combined_img = yolov8_detector.draw_detections(img)
    combined_img = yolov8_detector.blur_boxes(combined_img)
    im = Image.fromarray(combined_img)
    image_name = f"doc/img/detected_objects{i}.jpg"
    im.save(image_name)
    print(f"saved {image_name}")
