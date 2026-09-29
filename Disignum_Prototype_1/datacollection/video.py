import cv2
import os
import yt_dlp

# Step 1: Download YouTube video with yt_dlp
url = "https://www.youtube.com/watch?v=uKKvNqA9N20"
output_file = "video.mp4"

ydl_opts = {
    "format": "mp4",
    "outtmpl": output_file,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# Step 2: Extract frames (same as before)
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

video = cv2.VideoCapture(output_file)
count = 0
sec = 0

while True:
    video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # jump to each second
    success, frame = video.read()
    if not success:
        break
    cv2.imwrite(os.path.join(output_folder, f"frame_{count:05d}.jpg"), frame)
    count += 1
    sec += 1

video.release()
print(f"✅ Done! Extracted {count} frames into '{output_folder}'")
