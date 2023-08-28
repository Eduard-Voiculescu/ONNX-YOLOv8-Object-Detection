from yolov8 import YOLOv8
from PIL import Image
from numpy import asarray

import cv2
import glob

onnx_new_model_path = "privacy2-s1.onnx"
images_folder = "pic/*"
yolov8_detector = YOLOv8(onnx_new_model_path, conf_thres=0.2, iou_thres=0.3)

images = []
for f in glob.iglob(images_folder):
    print('file name', f)
    img = cv2.imread(f, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

for i, img in enumerate(images):
    boxes, scores, classe_ids = yolov8_detector(img)
    combined_img = yolov8_detector.draw_detections(img)
    combined_img = yolov8_detector.blur_boxes(img)
    # print(f'image bytes {combined_img.tobytes()}')
    im = Image.fromarray(combined_img)
    image_name = f"doc/img/detected_objects{i}.jpg"
    im.save(image_name)
    print(f"saved {image_name}")
