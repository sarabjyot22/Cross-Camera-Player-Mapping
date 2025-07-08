
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


broadcast_path = "../broadcast.mp4"
tacticam_path = "../tacticam.mp4"
model_path = "../yolov11_custom.pt"

model = YOLO(model_path)

tracker_broadcast = DeepSort(max_age=30)
tracker_tacticam = DeepSort(max_age=30)

def detect_and_track(video_path, tracker):
    cap = cv2.VideoCapture(video_path)
    frames_data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

        tracks = tracker.update_tracks(detections, frame=frame)
        frame_players = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltwh()
            frame_players[track_id] = bbox
        frames_data[cap.get(cv2.CAP_PROP_POS_FRAMES)] = frame_players

    cap.release()
    return frames_data

print("Processing broadcast.mp4 ...")
broadcast_data = detect_and_track(broadcast_path, tracker_broadcast)

print("Processing tacticam.mp4 ...")
tacticam_data = detect_and_track(tacticam_path, tracker_tacticam)

print("Sample Frame IDs from Broadcast Video:")
for frame_id, data in list(broadcast_data.items())[:5]:
    print(f"Frame {int(frame_id)}: Player IDs: {list(data.keys())}")

print("Sample Frame IDs from Tacticam Video:")
for frame_id, data in list(tacticam_data.items())[:5]:
    print(f"Frame {int(frame_id)}: Player IDs: {list(data.keys())}")
