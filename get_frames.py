import cv2
import os

# Path to the input video file
video_file = 'Data.avi'

# Create a directory for storing frames if it doesn't exist
output_dir = 'raw-frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Initialize variables
frame_count = 0
frame_rate = 30  # Number of seconds per frame

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Calculate the time in seconds
    time_in_seconds = frame_count / cap.get(cv2.CAP_PROP_FPS)

    # Check if it's time to save a frame (every 10 seconds)
    if time_in_seconds % frame_rate == 0:
        # Construct the output filename
        output_file = os.path.join(output_dir, f'frame_{frame_count}.jpg')

        # Save the frame as an image
        cv2.imwrite(output_file, frame)

    # Increment the frame count
    frame_count += 1

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()

print(f"Frames extracted and saved in '{output_dir}'")
