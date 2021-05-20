######################################################################################
### This is a read-only file to allow participants to run their code locally.      ###
### It will be over-writter during the evaluation, Please do not make any changes  ###
### to this file.                                                                  ###
######################################################################################

import json
import traceback
import os
import signal
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join

from evaluator import aicrowd_helpers


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Prediction timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class AirbornePredictor:
    def __init__(self):
        self.test_data_path = os.getenv("TEST_DATASET_PATH", os.getcwd() + "/data/val/")
        self.inference_output_path = self.get_results_directory()
        self.inference_setup_timeout = int(os.getenv("INFERENCE_SETUP_TIMEOUT_SECONDS", "600"))
        self.inference_flight_timeout = int(os.getenv("INFERENCE_PER_FLIGHT_TIMEOUT_SECONDS", "600"))
        self.partial_run = os.getenv("PARTIAL_RUN_FLIGHTS", None)
        self.results = []
        self.current_flight_results = []
        self.current_img_name = None
        self.track_id_seq = 0

    def register_object_and_location(self, class_name, track_id, bbox, confidence, img_name):
        """
        Register all your tracking results to this function.
           `track_id`: unique id for your detected airborne object
           `img_name`: The image_name to which this prediction belongs.
        """
        assert 0 < confidence < 1
        assert track_id is not None
        if img_name is None:
            img_name = self.current_img_name

        result = {
            "detections": [
                {
                    "track_id": track_id,
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                    "n": class_name,
                    "s": confidence
                }
            ],
            "img_name": img_name
        }
        self.results.append(result)
        self.current_flight_results.append(result)

    def get_all_flight_ids(self):
        valid_flight_ids = None
        if self.partial_run:
            valid_flight_ids = self.partial_run.split(',')
        flight_ids = []
        for folder in listdir(self.test_data_path):
            if not isfile(join(self.test_data_path, folder)):
                if valid_flight_ids is None or folder in valid_flight_ids:
                    flight_ids.append(folder)
        return flight_ids

    def get_all_frame_images(self, flight_id):
        frames = []
        flight_folder = join(self.test_data_path, flight_id)
        for frame in listdir(flight_folder):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames

    def get_frame_image_location(self, flight_id, frame_id):
        return join(self.test_data_path, flight_id, frame_id)

    def get_flight_folder_location(self, flight_id):
        return join(self.test_data_path, flight_id)

    def evaluation(self):
        """
        Admin function: Runs the whole evaluation
        """
        aicrowd_helpers.execution_start()
        try:
            with time_limit(self.inference_setup_timeout):
                self.inference_setup()
        except NotImplementedError:
            print("inference_setup doesn't exist for this run, skipping...")

        aicrowd_helpers.execution_running()

        flights = self.get_all_flight_ids()

        for flight in flights:
            with time_limit(self.inference_flight_timeout):
                self.inference(flight)
                
            self.save_results(flight)

        self.save_results()
        aicrowd_helpers.execution_success()

    def run(self):
        try:
            self.evaluation()
        except Exception as e:
            error = traceback.format_exc()
            print(error)
            aicrowd_helpers.execution_error(error)
            if not aicrowd_helpers.is_grading():
                raise e

    def inference_setup(self):
        """
        You can do any preprocessing required for your codebase here : 
            like loading your models into memory, etc.
        """
        raise NotImplementedError

    def inference(self, flight_id):
        """
        This function will be called for all the flight during the evaluation.
        NOTE: In case you want to load your model, please do so in `inference_setup` function.
        """
        raise NotImplementedError

    def get_results_directory(self, flight_id=None):
        """
        Utility function: results directory path
        """
        root_directory = os.getenv("INFERENCE_OUTPUT_PATH", os.getcwd() + "/data/results/")
        run_id = os.getenv("DATASET_ENV", "run0")
        results_directory = os.path.join(root_directory, run_id)
        if flight_id is not None:
            results_directory = os.path.join(results_directory, flight_id)
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        return results_directory

    def save_results(self, flight_id=None):
        """
        Utility function: save results in nested direcotry based on flight_id
        This helps in giving continuous feedback based on of the approximate 
        scores.
        """
        if flight_id is None:
            submission = self.results
        else:
            submission = self.current_flight_results

        with open(os.path.join(self.get_results_directory(flight_id), "result.json"), 'w') as fp:
            json.dump(submission, fp)

        self.current_flight_results = []
