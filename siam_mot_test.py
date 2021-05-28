# This file is the entrypoint for your submission.
# You can modify this file to include your code or directly call your functions/modules from here.
import random
import cv2
from PIL import Image
from evaluator.airborne_detection import AirbornePredictor

import numpy as np
from tqdm import tqdm

import os
from os import listdir
from os.path import isfile, join

from siam_mot_tracker import SiamMOTTracker

class SiamMOTPredictor(AirbornePredictor):
    """
    PARTICIPANT_TODO: You can name your implementation as you like. `RandomPredictor` is just an example.
    Below paths will be preloaded for you, you can read them as you like.
    """
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    """
    PARTICIPANT_TODO:
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def inference_setup(self):
        current_path = os.getcwd()
        config_file = os.path.join(current_path, 'siam-mot/configs/dla/DLA_34_FPN_AOT.yaml')
        model_path = os.path.join(current_path, 'siam-mot/models/DLA-34-FPN_box_track_aot_d4.pth')
        self.siammottracker = SiamMOTTracker(config_file, model_path)

    def get_all_frame_images(self, flight_id):
        frames = []
        flight_folder = join(self.test_data_path, flight_id)
        for frame in sorted(listdir(flight_folder)):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames

    """
    PARTICIPANT_TODO:
    During the evaluation all combinations for flight_id and flight_folder_path will be provided one by one.
    """
    def inference(self, flight_id):
        self.siammottracker.tracker.reset_siammot_status()

        for frame_image in tqdm(self.get_all_frame_images(flight_id)):
            frame_image_path = self.get_frame_image_location(flight_id, frame_image)
            frame = cv2.imread(frame_image_path)
            
            results = self.siammottracker.process(frame)

            class_name = 'airborne'
            for idx in range(len(results.bbox)):
                confidence = results.get_field('scores')[idx]
                if confidence < 0.3: # filter low confidence predictions
                    continue

                bbox_xywh = results.bbox.cpu().numpy()[idx]

                # bbox needed is [x0, y0, x1, y1] (top, left, bottom, right)
                bbox = [ float(bbox_xywh[0]), float(bbox_xywh[1]),
                         float(bbox_xywh[0] + bbox_xywh[2]),
                         float(bbox_xywh[1] + bbox_xywh[3])]

                track_id = results.get_field('ids')[idx]

                self.register_object_and_location(class_name, int(track_id), 
                                                bbox, float(confidence), 
                                                frame_image)

if __name__ == "__main__":
    submission = SiamMOTPredictor()
    submission.run()
