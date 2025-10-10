import cv2
import os

# === Configuration ===
video_path = "VID20250923092929.mp4"      # Path to your input video
output_folder = "Frames"     # Folder where frames will be saved

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Video FPS: {fps}")
print(f"Total Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save each frame as an image file
    frame_filename = os.path.join(output_folder, f"frame_{frame_index:04d}.png")
    cv2.imwrite(frame_filename, frame)

    frame_index += 1

    # Optional: progress display
    if frame_index % 100 == 0:
        print(f"Saved {frame_index}/{frame_count} frames...")

cap.release()
print(f"✅ Done! Extracted {frame_index} frames to '{output_folder}'")
