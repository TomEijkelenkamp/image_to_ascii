import cv2

# Parameters
input_path = "matrix_silhouette_output_6.mp4"
output_path = "matrix_silhouette_output_6_looped.mp4"
loop_count = 30  # how many times to repeat the video

# Open video
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

# Read all frames into memory
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Write repeated frames
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for _ in range(loop_count):
    for frame in frames:
        out.write(frame)

out.release()
print(f"Saved: {output_path}")
