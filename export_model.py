from ultralytics import YOLO
model = YOLO("yolo11n.pt")        # 3.1 MB checkpoint
model.export(format="ncnn", half=True, imgsz=320)
print("NCNN model saved in yolo11n_ncnn_model/")
