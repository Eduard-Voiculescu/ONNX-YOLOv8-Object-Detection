from yolov8 import YOLOv8
from PIL import Image
from numpy import asarray

import cv2
import io
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
            print("event", event)

            if type_names[0] == 'IN_CREATE':
                new_folder_path = os.path.join(path, name)
                if os.path.isdir(new_folder_path) and name[0] != r'_':
                    print(f'Processing new folder {new_folder_path}')
                    self._process_folder(name, new_folder_path)
                    print("byeeeee")
                
                if os.path.isdir(new_folder_path) and 'completed' in name:
                    print(f'completed folder {name}, pack, delete and write to /mnt/data/framekm path')

    def _process_folder(self, name: str, new_folder_path: str):
        framekm_name = os.path.join(new_folder_path, f'bin_{name}') 
        images = []
        print("folder path", new_folder_path)
        print(os.listdir(new_folder_path))
        for f in os.listdir(new_folder_path):
            p = os.path.join(new_folder_path, f)
            print('file name', p)
            img = cv2.imread(p, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        with open(f'{framekm_name}', 'ab') as f:
            metadata = ml_metadata.MLMetadata()  # for all the frames in a packed framekm
            for img in images:
                img_ml_data = ml_metadata.MLData()  # for all the boxes of an image

                boxes, scores, classe_ids = self.onnx_detector(img)
                combined_img = self.onnx_detector.draw_detections(img)
                combined_img = self.onnx_detector.blur_boxes(img)
                
                im = Image.fromarray(combined_img)
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                f.write(img_byte_arr)
        
        # os.rename(new_folder_path, f'completed_{path}')
