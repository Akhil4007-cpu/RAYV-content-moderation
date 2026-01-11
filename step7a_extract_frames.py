"""
Step 7a: Extract Frames from Video
Extracts frames from input video at specified intervals
"""
import cv2
import os
import sys
import argparse
import config

# Parse command line arguments
parser = argparse.ArgumentParser(description="Extract frames from video")
parser.add_argument("--video", type=str, default="input_video.mp4", help="Path to input video file")
args = parser.parse_args()

VIDEO_PATH = args.video
OUT_DIR = config.FRAMES_DIR_RAW
FPS_SAMPLE = config.FPS_SAMPLE_RATE

if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video file not found: {VIDEO_PATH}")
    print("   Please provide a valid video file using --video argument")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"📹 Extracting frames from {VIDEO_PATH}...")
print(f"   Output directory: {OUT_DIR}")
print(f"   Sample rate: {FPS_SAMPLE} seconds")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error opening video file: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"   Video FPS: {fps:.2f}")
print(f"   Total frames: {total_frames}")
print(f"   Duration: {duration:.2f} seconds")

if fps <= 0:
    print("❌ Invalid FPS detected")
    cap.release()
    sys.exit(1)

interval = max(1, int(fps * FPS_SAMPLE))

frame_id = 0
saved = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            frame_path = os.path.join(OUT_DIR, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
            
            if saved % 10 == 0:
                print(f"   Extracted {saved} frames...", end='\r')

        frame_id += 1

    cap.release()
    print(f"\n✅ Successfully extracted {saved} frames to {OUT_DIR}")
    
except Exception as e:
    print(f"\n❌ Error during frame extraction: {e}")
    cap.release()
    sys.exit(1)
