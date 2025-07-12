import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract cameras
cameras = config['organizations']['']['ip_cameras']

# Output folders
normal_folder = 'normal/'
alert_folder = 'alert/'
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(alert_folder, exist_ok=True)

# Load models
gun_model = YOLO('best_m_gun.pt')  # Detects guns (class 0 = gun)
uniform_model = YOLO('best.pt')  # class 0 = uniform, class 1 = normal dress

# Helper: Calculate IoU
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Helper: Draw bounding boxes
def draw_boxes(image, detections, label, color=(0, 255, 0)):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Build RTSP URL from config
def build_rtsp_url(ip, port, username, password):
    return f"rtsp://{username}:{password}@{ip}:{port}/"

# Process each camera
for cam_id, cam_info in cameras.items():
    rtsp_url = build_rtsp_url(
        cam_info['ip'],
        cam_info['port'],
        cam_info['username'],
        cam_info['password']
    )
    location = cam_info['location']
    branch_id = cam_info['branch_id']
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open camera: {cam_id} ({rtsp_url})")
        continue
    print(f"Processing camera: {cam_id} at {location}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from {cam_id}")
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue  # Skip alternate frame

        img = frame.copy()

        # Detect guns
        gun_results = gun_model(img)[0]
        gun_dets = [box.cpu().numpy() for box in gun_results.boxes.data if int(box[5]) == 0]

        if not gun_dets:
            continue  # No gun → skip

        # Detect people (uniformed and normal dress)
        uniform_results = uniform_model(img)[0]
        uniform_dets = [box.cpu().numpy() for box in uniform_results.boxes.data if int(box[5]) == 0]  # class 0 = uniform
        normal_dets = [box.cpu().numpy() for box in uniform_results.boxes.data if int(box[5]) == 1]   # class 1 = normal dress

        # Match guns to normal-dressed people
        alert_flag = False
        overlapped_normals = []
        for gun in gun_dets:
            gun_box = gun[:4]
            for person in normal_dets:
                person_box = person[:4]
                iou = calculate_iou(gun_box, person_box)
                if iou > 0.01:
                    alert_flag = True
                    overlapped_normals.append(person)

        # Save image accordingly
        filename = f"{cam_id}_{frame_count}.jpg"
        if alert_flag:
            img_annotated = img.copy()
            draw_boxes(img_annotated, gun_dets, label='Gun', color=(0, 0, 255))
            draw_boxes(img_annotated, overlapped_normals, label='Militant', color=(0, 255, 255))
            cv2.imwrite(os.path.join(alert_folder, filename), img_annotated)
        else:
            img_annotated = img.copy()
            draw_boxes(img_annotated, gun_dets, label='Gun', color=(0, 0, 255))
            draw_boxes(img_annotated, uniform_dets, label='Uniform', color=(0, 255, 0))
            cv2.imwrite(os.path.join(normal_folder, filename), img_annotated)

    cap.release()
