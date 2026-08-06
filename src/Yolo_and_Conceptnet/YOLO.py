import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from Yolo_and_Conceptnet.conceptnet import get_info

class YOLOPipeline:

    def __init__(self):

        self.model = YOLO("../yolo26l-seg.pt")  # segmentation helps noisy bounding box results
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.conceptnet_cache = {}


    def build_detection(self, label, confidence_score, x1, y1, x2, y2):
        return {
            "label": label,
            "confidence": confidence_score,
            "centre_x": int((x1 + x2) / 2),
            "centre_y": int((y1 + y2) / 2),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "mask": None
        }  # keep as fallback for bad masks and visualisation

    def build_segmentation(self, label, confidence_score, centre_x, centre_y, bbox, mask):
        return {
            "label": label,
            "confidence": confidence_score,
            "centre_x": int(centre_x),
            "centre_y": int(centre_y),
            "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            "mask": mask
        }  # main method

    def runYolo(self, frame):
        with torch.no_grad():
            detection_results = self.model(frame, conf=0.5, verbose=False)
        return detection_results

    def processDetections(self, detection_results, frame):
        detections_in_frame = []

        for result in detection_results:
            boxes = result.boxes
            masks = result.masks

            for i in range(len(boxes)):
                box = boxes[i]  # iterate through boxes
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                object_label = self.model.names[int(box.cls[0])]
                confidence_score = float(box.conf[0])

                detection = None  # initialise here
                resized_mask = None

                if masks is not None:
                    mask = masks.data[i].cpu().numpy()  # iterate through masks, put to cpu for numpy
                    resized_mask = cv2.resize(
                        mask,
                        (frame.shape[1], frame.shape[0])
                    )  # resize to frame display, there's a mismatch between yolo and display

                    y, x = np.where(resized_mask > 0)  # where there is a pixel value

                    if len(x) > 0 and len(y) > 0:  # and if the mask has worked
                        bbox = (x1, y1, x2, y2)
                        centre_x = x.mean()  # calculate centroid based on mask
                        centre_y = y.mean()

                        detection = self.build_segmentation(
                            object_label,
                            confidence_score,
                            centre_x,
                            centre_y,
                            bbox,
                            resized_mask
                        )
                    else:
                        detection = self.build_detection(
                            object_label,
                            confidence_score,
                            x1, y1, x2, y2
                        )  # mask fallback
                else:
                    detection = self.build_detection(
                        object_label,
                        confidence_score,
                        x1, y1, x2, y2
                    )  # mask fallback

                detections_in_frame.append(detection)

        return detections_in_frame

    def conceptnetInfo(self, object_label):
        if object_label not in self.conceptnet_cache:
            print(f"[ConceptNet] Looking up '{object_label}'...")
            self.conceptnet_cache[object_label] = get_info(object_label)

        concept_info = self.conceptnet_cache[object_label]
        return concept_info # useful for inference downstream

    def drawDetections(self, frame_display, detections_in_frame):
        for detection in detections_in_frame:
            x1, y1, x2, y2 = detection["bbox"]
            object_label = detection["label"]
            confidence_score = detection["confidence"]
            mask = detection["mask"]

            concept_info = self.conceptnetInfo(object_label)

            cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if mask is not None:
                mask_binary = (mask > 0).astype(np.uint8)  # draw mask contours instead of rectangle
                colour = np.array([0, 0, 255], dtype=np.uint8)
                coloured_mask = np.zeros_like(frame_display)
                coloured_mask[mask_binary == 1] = colour
                frame_display = cv2.addWeighted(frame_display, 1.0, coloured_mask, 0.5, 0)

            cv2.putText(
                frame_display,
                f"{object_label} {confidence_score:.0%}",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame_display,
                (detection["centre_x"], detection["centre_y"]),
                5,
                (0, 0, 255),
                -1
            )

            if concept_info: # if not none or empty dict
                relation = list(concept_info.keys())[0]
                fact = concept_info[relation][0]
                cv2.putText(
                    frame_display,
                    f"{relation}: {fact}",
                    (int(x1), int(y2) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 255),
                    1
                )

        return frame_display




