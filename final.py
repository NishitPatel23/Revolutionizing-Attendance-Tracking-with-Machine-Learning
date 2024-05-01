from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("bikes.mp4")  # For Video


model = YOLO("FinalModel.pt")

classNames = ['22IT100', '22IT104', '22IT105', '22IT108', '22IT110', '22IT112', '22IT113', '22IT114', '22IT115', '22IT116', 'charusat-idcard']
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)