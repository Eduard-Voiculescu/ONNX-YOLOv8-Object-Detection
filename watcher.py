from yolov8 import YOLOv8
from PIL import Image
from numpy import asarray

import cv2
import glob
import inotify.adapters
import inotify.constants
import ml_metadata
import os


class Watcher:
    notifier: inotify.adapters.Inotify
    onnx_detector: YOLOv8

    def __init__(self, detector: YOLOv8):
        self.notifier = inotify.adapters.Inotify()
        self.onnx_detector = detector

    def add_watch(self, path: str):
        self.notifier.add_watch(path)
    
    def run(self): 
        for event in self.notifier.event_gen(yield_nones=False):
            (_, type_names, path, name) = event
            print(event)

            if type_names[0] == 'IN_CREATE':
                new_folder_path = os.path.join(path, name)
                if os.path.isdir(new_folder_path) and 'completed' not in new_folder_path:
                    self._process_folder(name, new_folder_path)

    def _process_folder(self, name: str, path: str):
        framekm_name = f'bin_{name}'
        images = []
        for f in glob.iglob(path):
            print('file name', f)
            img = cv2.imread(f, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        with open(f'{framekm_name}', 'r+b') as f:
            metadata = ml_metadata.MLMetadata()
            for i, img in enumerate(images):
                img_ml_data = ml_metadata.MLData()

                boxes, scores, classe_ids = self.onnx_detector(img)
                combined_img = self.onnx_detector.draw_detections(img)
                combined_img = self.onnx_detector.blur_boxes(img)
                f.write(combined_img.tobytes())
                # im = Image.fromarray(combined_img)
                # image_name = f"doc/img/detected_objects{i}.jpg"
                # im.save(image_name)
                # print(f"saved {image_name}")
        
        os.rename(path, f'completed_{path}')
