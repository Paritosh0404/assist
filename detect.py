import cv2, time
from ultralytics import YOLO

class Detector:
    def __init__(self, cam=0, model_dir="yolo11n_ncnn_model"):
        self.cam = cv2.VideoCapture(cam)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.model = YOLO(model_dir)            # Loads NCNN weights
    def grab(self):
        _, frame = self.cam.read()
        return frame
    def infer(self, frame, conf=0.45):
        results = self.model(frame, imgsz=320, conf=conf, device="cpu")[0]
        return [(r.cls, r.xyxy.cpu(), r.conf) for r in results]
