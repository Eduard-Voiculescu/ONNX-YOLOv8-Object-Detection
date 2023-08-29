from yolov8 import YOLOv8
from PIL import Image

import constant
import cv2
import io
import inotify.adapters
import inotify.constants
import logging
import ml_metadata
import os
import shutil


class Watcher:
    notifier: inotify.adapters.Inotify
    onnx_detector: YOLOv8
    logger: logging.Logger

    def __init__(self, detector: YOLOv8, logger: logging.Logger):
        self.notifier = inotify.adapters.Inotify()
        self.onnx_detector = detector
        self.logger = logger

    def add_watch(self, path: str):
        self.notifier.add_watch(path)
    
    def run(self): 
        for event in self.notifier.event_gen(yield_nones=False):
            (_, type_names, path, name) = event
            self.logger.debug(f'[Event] type_names: {type_names}, path: {path}, name: {name}')
            
            if type_names[0] == 'IN_CREATE' or type_names[0] == "IN_MOVED_TO":
                new_path = os.path.join(path, name)
                if os.path.isdir(new_path) and name[0] != r'_':
                    self.logger.info(f"processing new folder: {new_path}")
                    self._process_folder(name, path, new_path)
                    self.logger.debug(f"done processing folder: {new_path}")
                
                if os.path.isfile(new_path) and 'completed_' in name:
                    renamed_path = os.path.join(constant.FRAMEKM, name.replace('completed_', ''))
                    self.logger.debug(f'path file {path}')
                    self.logger.debug(f'new_path {new_path}')
                    self.logger.debug(f'renamed_path {renamed_path}')
                    os.rename(new_path, renamed_path)
                    self.logger.info(f'{new_path} moved to {renamed_path}')

    def _process_folder(self, name: str, orig_path: str, new_folder_path: str):
        framekm_name = os.path.join(new_folder_path, f'bin_{name}') 
        images = []
        for f in os.listdir(new_folder_path):
            p = os.path.join(new_folder_path, f)
            self.logger.debug(f'file name {p}')
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
        
        self.logger.info(f'moving and cleaning up directories and files for {name}')
        framkm_name_orig_path = os.path.join(orig_path, f'completed_{name}')
        os.rename(framekm_name, framkm_name_orig_path)
        shutil.rmtree(new_folder_path)
