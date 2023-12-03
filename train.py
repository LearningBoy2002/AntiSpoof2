from ultralytics import YOLO

model = YOLO('yolov8m.pt')

def main():
    model.train(data='data.yaml', epochs=10)


if __name__ == '__main__':
    main()