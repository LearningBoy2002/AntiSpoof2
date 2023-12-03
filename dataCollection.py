import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time
import os

classID = 1  # 0 is fake and 1 is real
outputFolderPath = 'Dataset1'
confidence = 0.8
save = True
blurThreshold = 35  # Larger is more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

# cap = cv2.VideoCapture(0)
# cap.set(3, camWidth)
# cap.set(4, camHeight)
#cap = cv2.VideoCapture('Videos/real3.mp4')# for Video
folder_path = "dataset/val/real"
detector = FaceDetector(minDetectionCon=0.5, #modelSelection=0
                        )

for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path, filename))
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)
    
    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file
    if bboxs:
           
        for bbox in bboxs:
                
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]
                # print(x, y, w, h)

                if score > confidence:

                    offsetW = (offsetPercentageW / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)
                    offsetH = (offsetPercentageH / 100) * h
                    y = int(y - offsetH * 3)
                    h = int(h + offsetH * 3.5)

#               avoiding below zero values
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if w < 0: w = 0
                    if h < 0: h = 0

                    #Blurriness 
                    imgFace = img[y:y + h, x:x + w]
                    cv2.imshow("Face", imgFace)
                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    if blurValue > blurThreshold:
                        listBlur.append(True)
                    else:
                        listBlur.append(False)

                    # ------  Normalize Values  --------
                    ih, iw, _ = img.shape
                    xc, yc = x + w / 2, y + h / 2

                    xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                    wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                    # print(xcn, ycn, wn, hn)

                    # avoiding values above 1 
                    if xcn > 1: xcn = 1
                    if ycn > 1: ycn = 1
                    if wn > 1: wn = 1
                    if hn > 1: hn = 1

                    listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                    # Drawing 
                    cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                    scale=2, thickness=3)
                    if debug:
                        cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                        cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                        scale=2, thickness=3)

        #Saving
            if save:
                if all(listBlur) and listBlur != []:
                    # Save Image
                    timeNow = time()
                    timeNow = str(timeNow).split('.')
                    timeNow = timeNow[0] + timeNow[1]
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                    #  Save Label Text File 
                    for info in listInfo:
                        f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                        f.write(info)
                        f.close()
                
        
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

img.release()
cv2.destroyAllWindows()