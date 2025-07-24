import time
from detect import Detector
from range_sensor import distance_cm
from tts import speak

det = Detector()
last_alert = 0

# Mapping YOLO class indices → friendly labels
LABELS = {0: "person", 56: "chair", 63: "laptop"}

while True:
    frame = det.grab()
    for cls, box, conf in det.infer(frame):
        name = LABELS.get(int(cls), "object")
        # Basic spatial cue: centre X of bbox
        cx = (box[0][0] + box[0][2]) / 2
        loc = "left" if cx < 213 else "right" if cx > 426 else "ahead"
        alert = f"{name} {loc}"
        if time.time() - last_alert > 1.2:
            speak(alert)
            last_alert = time.time()

    # Proximity override
    d = distance_cm()
    if d < 80 and time.time() - last_alert > 0.8:
        speak(f"Obstacle {int(d)} centimetres")
        last_alert = time.time()
