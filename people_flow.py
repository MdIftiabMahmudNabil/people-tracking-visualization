import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# Delay for each frame display of OpenCV window
SLOWDOWN_MS = 50

# video path
VIDEO_PATH = 'data/people-walking.mp4'

# YOLO model
MODEL_PATH = 'yolov8n.pt'

# COCO class for person
PERSON_CLASS_ID = 0

# In line Coordinates
LINE_IN_START = (0, 302)
LINE_IN_END   = (1920, 302)
# Out line Coordinates
LINE_OUT_START = (0, 702)
LINE_OUT_END   = (1920, 702)

def main():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model from '{MODEL_PATH}'.")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Failed to open video file.')
        return

    ret, frame = cap.read()
    if not ret:
        print('Failed to read first frame.')
        return
    H, W = frame.shape[:2]

    # Save annotated output video
    os.makedirs("output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output/people_count_annotated.mp4", fourcc, 30, (W, H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    in_count = 0
    out_count = 0
    already_counted_in = set()
    already_counted_out = set()
    track_history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        boxes = results.boxes
        if boxes is None or boxes.xyxy is None:
            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('People Counting', display_frame)
            key = cv2.waitKey(SLOWDOWN_MS) & 0xFF
            if key == ord('q'):
                break
            continue
        class_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None]*len(xyxy)
        people_indices = [i for i, cid in enumerate(class_ids) if cid == PERSON_CLASS_ID]

        # Draw lines (green IN, red OUT)
        cv2.line(frame, LINE_IN_START, LINE_IN_END, (0,255,0), 2)
        cv2.line(frame, LINE_OUT_START, LINE_OUT_END, (0,0,255), 2)

        for idx in people_indices:
            x1, y1, x2, y2 = map(int, xyxy[idx])
            track_id = track_ids[idx]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(center)
            if len(track_history[track_id]) > 2:
                track_history[track_id] = track_history[track_id][-2:]
            if len(track_history[track_id]) == 2:
                prev_center = track_history[track_id][0]
                curr_center = track_history[track_id][1]
                # IN: crossing IN line downwards
                if prev_center[1] < LINE_IN_START[1] and curr_center[1] >= LINE_IN_START[1] and track_id not in already_counted_in:
                    in_count += 1
                    already_counted_in.add(track_id)
                # OUT: crossing OUT line upwards
                if prev_center[1] > LINE_OUT_START[1] and curr_center[1] <= LINE_OUT_START[1] and track_id not in already_counted_out:
                    out_count += 1
                    already_counted_out.add(track_id)
            # Draw boundingbox and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID {track_id}', (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.putText(frame, f'In: {in_count}', (20, LINE_IN_START[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)
        cv2.putText(frame, f'Out: {out_count}', (20, LINE_OUT_START[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        # Write annotated frame to video
        writer.write(frame)

        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('People Counting', display_frame)
        key = cv2.waitKey(SLOWDOWN_MS) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Total IN count: {in_count}")
    print(f"Total OUT count: {out_count}")
    print("Saved annotated video as output/people_count_annotated.mp4")

if __name__ == '__main__':
    main()
