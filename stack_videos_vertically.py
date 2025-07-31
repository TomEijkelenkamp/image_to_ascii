import cv2
import numpy as np

# Input files
video2_path = "clip_fixed_looped_30.mp4"
video1_path = "matrix_silhouette_output_2_looped.mp4"
output_path = "stack.mp4"

# Open both videos
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Get properties (assume both have same FPS)
fps = cap1.get(cv2.CAP_PROP_FPS)
width1  = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2  = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize bottom video to same width if needed
target_width = max(width1, width2)

# Output video settings
stacked_height = height1 + height2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, stacked_height))

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break  # Stop if either video ends

    # Resize frames to same width if necessary
    if frame1.shape[1] != target_width:
        frame1 = cv2.resize(frame1, (target_width, height1))
    if frame2.shape[1] != target_width:
        frame2 = cv2.resize(frame2, (target_width, height2))

    # Stack vertically
    stacked_frame = np.vstack((frame1, frame2))

    out.write(stacked_frame)

cap1.release()
cap2.release()
out.release()
print(f"Saved: {output_path}")
