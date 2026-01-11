"""
Step 7c: Build Temporal Windows
Groups frames into sliding temporal windows for analysis
"""
import os
import sys
import config

FRAMES_DIR = config.FRAMES_DIR_UNIQUE
WINDOW_SIZE = config.WINDOW_SIZE

if not os.path.exists(FRAMES_DIR):
    print(f"❌ Frames directory not found: {FRAMES_DIR}")
    print("   Please run step7a_extract_frames.py and step7b_remove_duplicates.py first")
    sys.exit(1)

frames = [f for f in sorted(os.listdir(FRAMES_DIR)) 
          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not frames:
    print(f"❌ No image files found in {FRAMES_DIR}")
    sys.exit(1)

if len(frames) < WINDOW_SIZE:
    print(f"❌ Not enough frames ({len(frames)}). Need at least {WINDOW_SIZE} frames for window analysis.")
    print(f"   Consider reducing WINDOW_SIZE in config.py or extracting more frames.")
    sys.exit(1)

windows = []
for i in range(0, len(frames) - WINDOW_SIZE + 1):
    windows.append(frames[i:i+WINDOW_SIZE])

print(f"✅ Created {len(windows)} temporal windows from {len(frames)} frames")
print(f"   Window size: {WINDOW_SIZE} frames")
print(f"   Overlap: {WINDOW_SIZE - 1} frames between consecutive windows")

# Save window information - use same directory as frames
output_dir = os.path.dirname(FRAMES_DIR) if os.path.dirname(FRAMES_DIR) else "."
windows_info_file = os.path.join(output_dir, "windows_info.json")
try:
    import json
    windows_info = {
        "total_frames": len(frames),
        "window_size": WINDOW_SIZE,
        "num_windows": len(windows),
        "windows": windows
    }
    with open(windows_info_file, 'w') as f:
        json.dump(windows_info, f, indent=2)
    print(f"   Window information saved to {windows_info_file}")
except Exception as e:
    print(f"   Warning: Could not save window info: {e}")
    # Fallback to root
    try:
        with open("windows_info.json", 'w') as f:
            json.dump(windows_info, f, indent=2)
    except:
        pass