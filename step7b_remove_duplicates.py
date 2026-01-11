"""
Step 7b: Remove Duplicate Frames
Uses CLIP to identify and remove visually similar frames
"""
import os
import sys
import torch
from PIL import Image
import config
from model_manager import get_model_manager

device = config.DEVICE

INPUT_DIR = config.FRAMES_DIR_RAW
OUTPUT_DIR = config.FRAMES_DIR_UNIQUE
SIM_THRESHOLD = config.CLIP_SIMILARITY_THRESHOLD

if not os.path.exists(INPUT_DIR):
    print(f"❌ Input directory not found: {INPUT_DIR}")
    print("   Please run step7a_extract_frames.py first")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🔍 Removing duplicate frames using CLIP...")
print(f"   Input directory: {INPUT_DIR}")
print(f"   Output directory: {OUTPUT_DIR}")
print(f"   Similarity threshold: {SIM_THRESHOLD}")

try:
    print(f"   Loading CLIP model ({device.upper()})...")
    manager = get_model_manager()
    processor, model = manager.get_clip()  # Uses shared model manager
    print("✅ CLIP model loaded (cached if already used)")
except Exception as e:
    print(f"❌ Error loading CLIP model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

prev_embed = None
kept = 0
skipped = 0
total = 0

image_files = [f for f in sorted(os.listdir(INPUT_DIR)) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"❌ No image files found in {INPUT_DIR}")
    sys.exit(1)

print(f"   Processing {len(image_files)} frames...")

try:
    for f in image_files:
        total += 1
        img_path = os.path.join(INPUT_DIR, f)
        
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                embed = model.get_image_features(**inputs)
                embed = embed / embed.norm(dim=-1, keepdim=True)

            if prev_embed is None:
                # Keep first frame
                output_path = os.path.join(OUTPUT_DIR, f)
                img.save(output_path)
                prev_embed = embed
                kept += 1
            else:
                similarity = torch.cosine_similarity(embed, prev_embed).item()
                if similarity < SIM_THRESHOLD:
                    # Frame is different enough, keep it
                    output_path = os.path.join(OUTPUT_DIR, f)
                    img.save(output_path)
                    prev_embed = embed
                    kept += 1
                else:
                    skipped += 1

            if total % 10 == 0:
                print(f"   Processed {total}/{len(image_files)} frames (kept: {kept}, skipped: {skipped})...", end='\r')

        except Exception as e:
            print(f"\n   Warning: Error processing {f}: {e}")
            continue

    print(f"\n✅ Successfully processed {total} frames")
    print(f"   Kept: {kept} unique frames")
    print(f"   Removed: {skipped} duplicate frames")
    print(f"   Reduction: {(skipped/total*100):.1f}%")
    
except Exception as e:
    print(f"\n❌ Error during duplicate removal: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
