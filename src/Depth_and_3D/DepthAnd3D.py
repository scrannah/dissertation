import torch
import numpy as np

class DepthPipeline:

    def __init__(self, fov_degrees=120.0, fov_type="diagonal", model_name="DPT_Large"):
        # camera field of view
        self.fov_degrees = fov_degrees
        self.fov_type = fov_type

        # load MiDaS model
        # hybrid might be better suited later if using time series
        self.midas = torch.hub.load("intel-isl/MiDaS", model_name)
        self.midas.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas = self.midas.to(self.device)

        # load MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

        # camera intrinsics, filled later once image size known
        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None

    def estimate_depth(self, img_rgb):
        # apply MiDaS transform
        input_batch = self.transform(img_rgb)
        input_batch = input_batch.to(self.device)

        # run model
        with torch.no_grad():
            pred = self.midas(input_batch)

        # resize prediction to original image size
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        # keep depth on GPU as a tensor, converting to numpy here forces every
        # downstream calculation onto CPU only convert to numpy at the point that needs it (cv2 display).
        return pred

    def process_image(self, image_rgb, detections, frame_id, timestamp):
        # pipeline
        depth = self.estimate_depth(image_rgb)

        # compute camera parameters once depth size known
        self.compute_intrinsics(depth)

        objects = [] # empty list to start from, each frame needs new object list

        # process each detection
        for detection in detections: # going over the detections in frame list
            result = self.process_detection(detection, depth) # for each detection on depth map, process
            objects.append(result) # append each detection result to object list

        scene_package = self.package_scene(frame_id, timestamp, objects) # package full scene as frame, with objects within

        print(scene_package)
        depth_cpu = depth.cpu().numpy() # converting to cpu for cv2 display
        return image_rgb, depth_cpu, scene_package # original frame, depth map, package


    def compute_intrinsics(self, depth):
        # get height and width
        h, w = depth.shape

        # diagonal length (pythagoras)
        d = np.sqrt(w**2 + h**2)

        # optical centre
        self.cx = w / 2.0
        self.cy = h / 2.0

        # convert FOV to radians
        fov = np.deg2rad(self.fov_degrees)

        # focal length calculation
        if self.fov_type == "horizontal" or self.fov_type == "vertical":

            self.fx = (w / 2.0) / np.tan(fov / 2.0)

        elif self.fov_type == "diagonal":

            self.fx = (d / 2.0) / np.tan(fov / 2.0)

        else:
            raise ValueError(f"Invalid fov_type '{self.fov_type}'. Must be 'horizontal', 'vertical', or 'diagonal'.") # for use in other camera models

        # assume square pixels
        self.fy = self.fx


    def process_detection(self, detection, depth):

        bbox = detection["bbox"]
        label = detection["label"]

        # get centre pixel of detection, remove bbox centre logic in depth it is unused
        u = detection["centre_x"]
        v = detection["centre_y"]

        mask = detection["mask"]

        # convert centre pixel to 3d, consider using median depth for bounding box
        # or use a cropped centre to remove background
        x, y, z = self.pixel_to_3d(u, v, depth, mask)

        # package just the object info
        result = self.package_object(label, x, y, z) # get object as a list

        return result

    def pixel_to_3d(self, u, v, depth, mask):
        if mask is None or np.sum(mask) == 0:
            # depth is now a GPU tensor, not a numpy array .item() pulls
            # the single scalar value back into a plain Python float
            z = depth[v, u].item()  # fallback to centre pixel
        else:
            # move mask to the same device as depth once, then index + median
            # entirely on GPU, numpy on cpu slows it down
            mask_gpu = torch.from_numpy(mask).to(self.device).bool()
            z = torch.median(depth[mask_gpu]).item()  # robust median over mask

        # convert pixel coordinates to camera coordinates
        # back-projection of pinhole camera model
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return x, y, z


    def package_object(self, label, x, y, z):
        # package one object for QSR
        return {
            "label": label,
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }


    def package_scene(self, frame_id, timestamp, objects):
        # package whole scene/frame for QSR
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects": objects
        }