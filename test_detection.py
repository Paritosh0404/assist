#!/usr/bin/env python3
import cv2
import time
from detect import Detector

# Complete COCO class names (80 classes)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}


def test_detection():
    print("Initializing detector...")
    detector = Detector(cam=0)  # USB camera
    print("Detector ready! Press 'q' to quit, 's' to save frame")

    frame_count = 0
    start_time = time.time()

    while True:
        # Capture frame
        frame = detector.grab()
        if frame is None:
            print("Failed to grab frame")
            continue

        # Run detection
        detection_start = time.time()
        detections = detector.infer(frame, conf=0.5)
        detection_time = time.time() - detection_start

        # Draw results
        for cls, bbox, conf in detections:
            cls_id = int(cls)
            class_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")

            # Extract coordinates
            x1, y1, x2, y2 = map(int, bbox[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Detected: {class_name} (confidence: {conf:.2f})")

        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Average FPS: {fps:.1f}, Detection time: {detection_time * 1000:.1f}ms")

        # Display frame
        cv2.imshow('Object Detection Test', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_test_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")

    # Cleanup
    cv2.destroyAllWindows()
    print("Detection test completed")


if __name__ == "__main__":
    test_detection()
