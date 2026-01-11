"""
Step 7d: Window Multimodal Analysis
Analyzes temporal windows using YOLO, RAM, and BLIP models
"""
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import config
from model_manager import get_model_manager
from risk_scorer import get_risk_scorer
from action_recognition import get_action_recognizer

device = config.DEVICE

# ==== LOAD MODELS (using shared ModelManager - no duplication!) ====
print(f"🔧 Loading models using shared ModelManager ({device.upper()})...")
try:
    manager = get_model_manager()
    yolo = manager.get_yolo()
    ram_model, ram_transform = manager.get_ram_model()
    blip_processor, blip_model = manager.get_blip()
    
    print(f"✅ All models loaded and cached (no reloading needed)")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==== SETTINGS ====
FRAMES_DIR = config.FRAMES_DIR_UNIQUE
WINDOW_SIZE = config.WINDOW_SIZE

if not os.path.exists(FRAMES_DIR):
    print(f"❌ Frames directory not found: {FRAMES_DIR}")
    print("   Please run step7a_extract_frames.py and step7b_remove_duplicates.py first")
    sys.exit(1)

# ==== HELPERS ====
def yolo_objects(image_path):
    """Extract objects from image using YOLO"""
    try:
        results = yolo(image_path, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            return [yolo.names[int(c)] for c in results.boxes.cls]
        return []
    except Exception as e:
        print(f"   Warning: YOLO error on {image_path}: {e}")
        return []

def ram_tags(image):
    """Extract scene tags using RAM"""
    try:
        image_tensor = ram_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            tags, tags_chinese = ram_model.generate_tag(image_tensor)
        # Handle return format - tags is a list, take first element
        tags_str = tags[0] if isinstance(tags, (list, tuple)) and len(tags) > 0 else str(tags) if tags else ""
        tags_chinese_str = tags_chinese[0] if isinstance(tags_chinese, (list, tuple)) and len(tags_chinese) > 0 else str(tags_chinese) if tags_chinese else ""
        return tags_str, tags_chinese_str
    except Exception as e:
        print(f"   Warning: RAM error: {e}")
        import traceback
        traceback.print_exc()
        return "", ""

def blip_caption(image):
    """Generate caption using BLIP"""
    try:
        inputs = blip_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=config.BLIP_MAX_TOKENS)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"   Warning: BLIP error: {e}")
        return ""

def motion_score(frame_objects, images):
    """Calculate motion score using multiple methods: frame differencing, optical flow, and object tracking"""
    if len(images) < 2:
        return 0.0
    
    # Method 1: Object count changes (baseline)
    obj_count_changes = []
    if len(frame_objects) > 1:
        for i in range(len(frame_objects) - 1):
            obj_count_changes.append(abs(len(frame_objects[i+1]) - len(frame_objects[i])))
    count_motion = float(np.mean(obj_count_changes)) if obj_count_changes else 0.0
    
    # Method 2: Frame differencing (improved)
    try:
        import cv2
        frame_diffs = []
        flow_magnitudes = []
        
        prev_gray = None
        for i in range(len(images)):
            # Convert PIL to numpy
            img = np.array(images[i])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            
            if prev_gray is not None:
                # Frame differencing
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff) / 255.0  # Normalize
                frame_diffs.append(mean_diff)
                
                # Optical flow (Lucas-Kanade method) - better motion detection
                try:
                    # Detect corners for optical flow
                    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
                    if corners is not None and len(corners) > 0:
                        # Calculate optical flow
                        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
                        
                        # Calculate flow magnitude
                        if next_pts is not None:
                            valid = status == 1
                            if np.any(valid):
                                flow = next_pts[valid] - corners[valid]
                                flow_mag = np.mean(np.linalg.norm(flow, axis=1))
                                flow_magnitudes.append(flow_mag / 100.0)  # Normalize
                except:
                    pass  # Skip optical flow if fails
            
            prev_gray = gray
        
        frame_motion = float(np.mean(frame_diffs)) if frame_diffs else 0.0
        optical_flow_motion = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        
        # Combine all methods (weighted average)
        # Optical flow is most reliable, frame diff is good, object count is baseline
        combined_motion = (
            count_motion * 0.1 +           # 10% weight (baseline)
            frame_motion * 10.0 * 0.4 +    # 40% weight (frame diff)
            optical_flow_motion * 10.0 * 0.5  # 50% weight (optical flow - best)
        )
        
        return combined_motion
    except Exception as e:
        # Fallback to original method if cv2 fails
        print(f"   Warning: Advanced motion detection failed, using object count: {e}")
        return count_motion

# ==== BUILD WINDOWS ====
frames = [f for f in sorted(os.listdir(FRAMES_DIR)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(frames) < WINDOW_SIZE:
    print(f"❌ Not enough frames ({len(frames)}). Need at least {WINDOW_SIZE} frames.")
    sys.exit(1)

windows = [
    frames[i:i+WINDOW_SIZE]
    for i in range(len(frames) - WINDOW_SIZE + 1)
]

print(f"🧠 Processing {len(windows)} windows...\n")

WINDOW_RESULTS = []

for idx, window in enumerate(windows):
    print(f"▶ Window {idx + 1}/{len(windows)}")

    window_objects = []
    frame_objects = []
    images = []

    # Load all frames in window
    for f in window:
        path = os.path.join(FRAMES_DIR, f)
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)

            objs = yolo_objects(path)
            frame_objects.append(objs)
            window_objects.extend(objs)
        except Exception as e:
            print(f"   Warning: Error processing {f}: {e}")
            continue

    if not images:
        print(f"   ⚠️  No valid images in window, skipping")
        continue

    # --- RAM (scene tags) - use middle frame (could batch process all frames for better accuracy) ---
    tags, tags_chinese = ram_tags(images[len(images)//2])
    
    # --- BLIP (caption) - use middle frame (could analyze multiple frames for better context) ---
    caption = blip_caption(images[len(images)//2])
    
    # Optional: Analyze multiple frames for better context (future improvement)
    # captions = [blip_caption(img) for img in images]
    # caption = most_common_caption(captions)  # Aggregate captions

    # --- MOTION --- (improved with frame differencing + optical flow)
    motion = motion_score(frame_objects, images)

    # --- AGGREGATION ---
    obj_counts = Counter(window_objects)
    main_objects = [k for k, v in obj_counts.items() if v >= 2]
    all_objects = list(obj_counts.keys())  # All unique objects
    
    # Count persons
    person_count = obj_counts.get("person", 0)
    
    # --- CONTEXT AWARENESS ---
    tags_lower = tags.lower() if tags else ""
    caption_lower = caption.lower() if caption else ""
    combined_context = f"{tags_lower} {caption_lower}"
    
    # Check if we're in a safe context (kitchen, cooking, etc.)
    is_safe_context = any(keyword in combined_context for keyword in config.SAFE_CONTEXT_KEYWORDS)
    
    # --- WEAPON DETECTION (FIXED) ---
    # Check actual YOLO classes first (most reliable)
    yolo_weapons_found = any(obj in config.WEAPON_YOLO_CLASSES for obj in all_objects)
    
    # Also check tags and captions for weapon mentions (less reliable but catches guns, etc.)
    tag_weapon_mentions = any(keyword in combined_context for keyword in config.WEAPON_KEYWORDS)
    
    weapon_detected = yolo_weapons_found or tag_weapon_mentions
    
    # --- ACTION RECOGNITION --- (detect specific violent actions)
    action_recognizer = get_action_recognizer()
    motion_list = [motion] * len(frame_objects)  # Approximate per-frame motion for action detection
    actions_detected = action_recognizer.detect_actions(frame_objects, motion_list, [caption], tags)
    action_risk = action_recognizer.calculate_action_risk(actions_detected) if actions_detected else 0.0
    
    # --- RISK LOGIC (LEARNED SCORER - ADVANCED) ---
    # Use learned risk scorer for better accuracy and feature engineering
    risk_scorer = get_risk_scorer()
    
    # Prepare window data for feature extraction
    window_data = {
        "weapon_detected": weapon_detected,
        "is_safe_context": is_safe_context,
        "person_count": person_count,
        "motion": motion,
        "caption": caption,
        "scene_tags": tags,
        "all_objects": all_objects,
        "main_objects": main_objects
    }
    
    # Extract features (feature engineering)
    features = risk_scorer.calculate_features(window_data)
    
    # Calculate risk score using learned scorer (replaces simple heuristics)
    risk, risk_factors = risk_scorer.score(features)
    
    # Add action-based risk (detected violent actions)
    if action_risk > 0:
        risk += action_risk * 0.3  # Weight action risk (30% of action confidence)
        risk_factors.append(f"action_detected({action_risk:.2f})")
        risk = min(risk, 1.0)  # Cap at 1.0
    
    label = (
        "VIOLENT" if risk >= 0.7 else
        "RISKY" if risk >= 0.3 else
        "SAFE"
    )

    result = {
        "window_id": idx,
        "frames": window,
        "objects": main_objects,
        "all_objects": all_objects,  # Track all objects
        "person_count": person_count,
        "weapon_detected": weapon_detected,
        "is_safe_context": is_safe_context,
        "actions_detected": actions_detected,  # New: detected actions
        "action_risk": round(action_risk, 2),  # New: action-based risk
        "scene_tags": tags,
        "scene_tags_chinese": tags_chinese,
        "caption": caption,
        "motion": round(motion, 2),
        "risk_score": round(risk, 2),
        "risk_factors": risk_factors,  # What contributed to risk
        "features": features,  # New: extracted features for analysis
        "label": label
    }

    WINDOW_RESULTS.append(result)

    print("   Objects:", main_objects[:5], "..." if len(main_objects) > 5 else "")
    print("   Caption:", caption[:60], "..." if len(caption) > 60 else "")
    print("   Motion:", round(motion, 2), "(frame diff + optical flow)")
    if weapon_detected:
        detected_weapons = [obj for obj in all_objects if obj in config.WEAPON_YOLO_CLASSES or any(w in obj.lower() for w in config.WEAPON_KEYWORDS)]
        print("   ⚠️  WEAPON DETECTED:", detected_weapons)
    if is_safe_context:
        print("   🏠 Safe context detected (kitchen/cooking)")
    if actions_detected:
        print("   ⚡ Actions detected:", ", ".join([f"{k}({v:.2f})" for k, v in actions_detected.items()]))
    print("   Risk factors:", ", ".join(risk_factors[:3]) if risk_factors else "none", "..." if len(risk_factors) > 3 else "")
    print("   Risk:", round(risk, 2), "→", label)
    print("-" * 40)

# Save results to JSON
output_file = config.WINDOW_RESULTS_FILE
try:
    with open(output_file, 'w') as f:
        json.dump(WINDOW_RESULTS, f, indent=2)
    print(f"\n✅ WINDOW ANALYSIS COMPLETE")
    print(f"✅ Results saved to {output_file}")
except Exception as e:
    print(f"❌ Error saving results: {e}")
    print("Results (not saved):")
    print(json.dumps(WINDOW_RESULTS[:3], indent=2))  # Show first 3 as sample
