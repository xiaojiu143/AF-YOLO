from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("AF-YOLO")
    model.train(data="your.yaml", epochs=500, batch=8, imgsz=640, patience=50, deterministic=False)
