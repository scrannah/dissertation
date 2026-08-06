
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.expanduser("~/Dissertation/src"))

import numpy as np

from Agent import Agent
from controller import Field, Node

from Yolo_and_Conceptnet import YOLOPipeline
# from Depth_and_3D import DepthPipeline  
# from QSR import QSRPipeline             


class RobotAgent(Agent):
    def __init__(self):
        Agent.__init__(self, agentName="Tiago++")

        self.yolo_pipeline = YOLOPipeline()
        # self.depth_pipeline = DepthPipeline()
        # self.qsr_pipeline = QSRPipeline()

        
        self.tracked_objects = {}   # {tracked_id: (x, y, label)}
        self.next_object_id = 0
        self.match_max_distance = 50  # pixels adjust if needed
        
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        self.rangefinder.enable(self.timestep)  

        print(str(self.__class__.__name__) + " has activated.")

    def get_camera_frame_as_array(self):
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        image = self.camera.getImage()

        frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        frame_bgr = frame[:, :, :3].copy()
        return frame_bgr
        
    def match_detections(self, current_detections):
        matched_ids = {}
        unmatched = list(current_detections)
    
        for tracked_id, tracked_data in list(self.tracked_objects.items()):
            prev_x, prev_y = tracked_data["pos"]
            prev_label = tracked_data["label"]
            best_match = None
            best_distance = self.match_max_distance
    
            for detection in unmatched:
                if detection["label"] != prev_label:
                    continue
                dx = detection["centre_x"] - prev_x
                dy = detection["centre_y"] - prev_y
                distance = (dx ** 2 + dy ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_match = detection
    
            if best_match:
                self.tracked_objects[tracked_id] = {
                    "pos": (best_match["centre_x"], best_match["centre_y"]),
                    "label": best_match["label"],
                    "missed_frames": 0
                }
                matched_ids[tracked_id] = best_match
                unmatched.remove(best_match)
            else:
                self.tracked_objects[tracked_id]["missed_frames"] += 1
                if self.tracked_objects[tracked_id]["missed_frames"] > 3:
                    del self.tracked_objects[tracked_id]
    
        newly_seen = []
        for detection in unmatched:
            new_id = self.next_object_id
            self.next_object_id += 1
            self.tracked_objects[new_id] = {
                "pos": (detection["centre_x"], detection["centre_y"]),
                "label": detection["label"],
                "missed_frames": 0
            }
            matched_ids[new_id] = detection
            newly_seen.append((new_id, detection["label"]))
    
        return matched_ids, newly_seen
    

    def perceive(self, debug=True):
        frame = self.get_camera_frame_as_array()

        detection_results = self.yolo_pipeline.runYolo(frame)
        detections_in_frame = self.yolo_pipeline.processDetections(detection_results, frame)

        tracked, newly_seen = self.match_detections(detections_in_frame)

        if debug:
            print(f"[PERCEIVE] {len(tracked)} tracked objects, {len(newly_seen)} newly seen")
            for tracked_id, detection in tracked.items():
                print(f"  id={tracked_id} label={detection['label']} "
                      f"pos=({detection['centre_x']}, {detection['centre_y']})")

        return tracked, newly_seen

    def generate_hypothesis(self, qsr_observations):
        raise NotImplementedError

    def update_belief(self, hypothesis, similarity_score):
        raise NotImplementedError

    def main(self):
        step_count = 0
        while self.step():
            step_count += 1

            if step_count % 30 == 0:
                tracked, newly_seen = self.perceive(debug=True)


def main():
    robot = RobotAgent()
    robot.main()


main()