import ultralytics
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import os
import time

model = YOLO('yolov8n.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker = Tracker()
count = 0

cap = cv2.VideoCapture("2.mp4")

down = {}
up = {}
counter_down = []
counter_up = []
speed_fixed_down = {}
speed_fixed_up = {}

red_line_y = 198
blue_line_y = 268
offset = 6

if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    car_list = []

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row[:6])
        if class_list[d] == 'car':
            car_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(car_list)

    for x3, y3, x4, y4, id in bbox_id:
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if red_line_y - offset < cy < red_line_y + offset:
            down[id] = time.time()
        
        if id in down and id not in speed_fixed_down:
            if blue_line_y - offset < cy < blue_line_y + offset:
                elapsed_time = time.time() - down[id]
                distance = 20
                speed_kh = (distance / elapsed_time) * 3.6
                speed_fixed_down[id] = speed_kh
        
        if id in speed_fixed_down:
            cv2.putText(frame, f'{int(speed_fixed_down[id])} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if blue_line_y - offset < cy < blue_line_y + offset:
            up[id] = time.time()
        
        if id in up and id not in speed_fixed_up:
            if red_line_y - offset < cy < red_line_y + offset:
                elapsed1_time = time.time() - up[id]
                distance1 = 20
                speed_kh1 = (distance1 / elapsed1_time) * 3.6
                speed_fixed_up[id] = speed_kh1
        
        if id in speed_fixed_up:
            cv2.putText(frame, f'{int(speed_fixed_up[id])} Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    cv2.line(frame, (172, 198), (774, 198), (0, 0, 255), 2)
    cv2.putText(frame, 'Red Line', (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.line(frame, (8, 268), (927, 268), (255, 0, 0), 2)
    cv2.putText(frame, 'Blue Line', (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, f'Going Down - {len(speed_fixed_down)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Going Up - {len(speed_fixed_up)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)
    out.write(frame)
    
    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
