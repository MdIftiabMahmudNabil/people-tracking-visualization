import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os

VIDEO_PATH = 'data/people-walking.mp4'
MODEL_PATH = 'yolov8n.pt'

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Failed to open video file.')
        return

    # Get frame size
    ret, frame = cap.read()
    if not ret:
        print('Failed to read first frame.')
        return
    H, W = frame.shape[:2]

    # Prepare VideoWriter
    os.makedirs("output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output/people_heatmap_overlay.mp4", fourcc, 30, (W, H))

    # Initialize Supervision HeatMapAnnotator
    heatmap_annotator = sv.HeatMapAnnotator(
        opacity=0.6,
        radius=40,
        kernel_size=25,
        top_hue=0,      # red
        low_hue=120     # blue
    )

    # Rewind video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for people (class_id == 0)
        mask = detections.class_id == 0
        detections.xyxy = detections.xyxy[mask]
        detections.confidence = detections.confidence[mask]
        detections.class_id = detections.class_id[mask]

        # Annotate heatmap
        frame_with_heatmap = heatmap_annotator.annotate(scene=frame.copy(), detections=detections)

        # Write overlay to video
        writer.write(frame_with_heatmap)

        # Resize to annotation size for display
        display_frame = cv2.resize(frame_with_heatmap, (960, 540))
        cv2.imshow('Supervision Heatmap', display_frame)
        # Fastest playback
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Processed {frame_count} frames...')

    # Save and display final static heatmap
    try:
        final_heatmap = heatmap_annotator.heatmap
    except AttributeError:
        final_heatmap = heatmap_annotator.heat_mask
    final_heatmap_norm = cv2.normalize(final_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    final_heatmap_color = cv2.applyColorMap(final_heatmap_norm, cv2.COLORMAP_JET)

    # Save final static heatmap as PNG
    cv2.imwrite('output/people_flow_heatmap.png', final_heatmap_color)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Saved overlay video as output/people_heatmap_overlay.mp4")
    print("Saved static heatmap as output/people_flow_heatmap.png")

if __name__ == '__main__':
    main()
