import math
from ultralytics import YOLO
import cv2
import cvzone
import torch

cap = cv2.VideoCapture(0)# for Webcam
cap.set(3,1280)
cap.set(4, 720)
max_cls_index = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = YOLO("yolov8m.pt")

classNames = ['real', 'spoof']

while True: 
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w,h = x2-x1 ,y2-y1
            # x1,y1,w,h = box.xywh[0]
            # bbox =  int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,(x1,y1,w,h))
            cls = int(box.cls[0])
            max_cls_index = max(max_cls_index, cls)
            conf = math.ceil((box.conf[0]*100))/100
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))
            print(conf)
            if 0 <= cls < len(classNames):
                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(0,y1-20)), scale=0.7, thickness=1)
            else:
                print(f"Invalid class index: {cls}")

    if len(classNames) <= max_cls_index:
        classNames.extend([''] * (max_cls_index + 1 - len(classNames)))


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# Destroy all the windows
cv2.destroyAllWindows()