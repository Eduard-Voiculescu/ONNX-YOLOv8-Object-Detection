from yolov8 import YOLOv8
from PIL import Image

import constant
import cv2
import json
import io
import inotify.adapters
import inotify.constants
import logging
import ml_metadata
import yolov8.utils
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
                
                if os.path.isfile(new_path) and 'km_completed_' in name:
                    renamed_path = os.path.join(constant.FRAMEKM, name.replace('km_completed_', ''))
                    self.logger.debug(f'path file {path}')
                    self.logger.debug(f'new_path {new_path}')
                    self.logger.debug(f'renamed_path {renamed_path}')
                    os.rename(new_path, renamed_path)
                    self.logger.info(f'{new_path} moved to {renamed_path}')
                
                if os.path.isfile(new_path) and 'metadata_ml_completed_' in name:
                    self.logger.info(f'completed ml {name}')
                    renamed_path = os.path.join(constant.ML_FRAMEKM_METADATA_JSON, name.replace('metadata_ml_completed_', '') + '.json')
                    os.rename(new_path, renamed_path)
                    self.logger.info(f'{new_path} moved to {renamed_path}')

    def _process_folder(self, name: str, orig_path: str, new_folder_path: str):
        framekm_name = os.path.join(new_folder_path, f'bin_{name}') 
        frames = []
        for f in os.listdir(new_folder_path):
            p = os.path.join(new_folder_path, f)
            self.logger.debug(f'file name {p}')
            img = cv2.imread(p, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append((p, img))
        
        with open(f'{framekm_name}', 'ab') as f:
            metadata = ml_metadata.MLMetadata()  # for all the frames in a packed framekm
            for val in frames:
                img_id = val[0]
                img = val[1]

                img_ml_data = ml_metadata.MLFrameData()  # for all the boxes of an image
                
                boxes, scores, classe_ids = self.onnx_detector(img, img_ml_data)
                converted_box = yolov8.utils.xyxyxywh2(boxes)
                for i, class_id in enumerate(classe_ids):
                    bounding_box = ml_metadata.BoundingBox()
                    bounding_box.set_class_id(constant.CLASS_NAMES[class_id])
                    bounding_box.set_confidence(scores[i])
                    bounding_box.set_cxcywh(converted_box[i])
                    img_ml_data.detections.append(bounding_box)

                combined_img = self.onnx_detector.draw_detections(img)
                combined_img = self.onnx_detector.blur_boxes(img, img_ml_data)
                
                im = Image.fromarray(combined_img)
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                f.write(img_byte_arr)

                img_ml_data.img_id = img_id
                img_ml_data.name = f'{name}.json'

                metadata.ml_frame_data.append(img_ml_data)

            metadata_path = os.path.join(orig_path, f'metadata_ml_completed_{name}')
            with open(metadata_path, 'w') as f:
                f.write(metadata.toJson())
        
        # TODO: write metadata to disk
        self.logger.info(f'moving and cleaning up directories and files for {name}')
        framkm_name_orig_path = os.path.join(orig_path, f'km_completed_{name}')
        os.rename(framekm_name, framkm_name_orig_path)
        shutil.rmtree(new_folder_path)
