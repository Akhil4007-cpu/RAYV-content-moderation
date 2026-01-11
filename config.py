"""
Configuration file for Vision AI Pipeline
"""
import os
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
YOLO_MODEL_PATH = "yolov8n.pt"
RAM_CHECKPOINT_PATH = "checkpoints/ram_swin_large_14m.pth"
RAM_IMAGE_SIZE = 384
RAM_VIT_TYPE = "swin_l"

# BLIP model
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
BLIP_MAX_TOKENS = 40

# CLIP model
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Video processing
FRAMES_DIR_RAW = "frames_raw"
FRAMES_DIR_UNIQUE = "frames_unique"
FPS_SAMPLE_RATE = 0.5  # seconds between frames

# Duplicate removal
CLIP_SIMILARITY_THRESHOLD = 0.97

# Temporal window analysis
WINDOW_SIZE = 5

# Risk assessment thresholds
RISK_MOTION_THRESHOLD = 1.2
RISK_WEAPON_WEIGHT = 0.6  # Increased weight for actual weapon detection
RISK_PERSON_MOTION_WEIGHT = 0.3  # Reduced - motion detection is less reliable
RISK_CAPTION_WEIGHT = 0.4
RISK_CONTEXT_PENALTY = 0.5  # Reduce risk in safe contexts (kitchen, etc.)
RISK_MULTIPLE_PERSONS_WEIGHT = 0.3  # Multiple people = higher risk
RISK_WEAPON_IN_CONTEXT_BONUS = 0.2  # Weapon in non-kitchen context = higher risk

# Decision engine thresholds
ESCALATION_THRESHOLD = 0.25
VIOLENT_RATIO_THRESHOLD = 0.25
RISKY_RATIO_THRESHOLD = 0.20

# Output paths (will be set per session)
OUTPUT_BASE_DIR = "outputs"
WINDOW_RESULTS_FILE = "window_results.json"

# Session management
CLEAN_PREVIOUS_RESULTS = False  # Set to True to auto-clean before processing
KEEP_RAW_FRAMES = True  # Keep raw frames after processing (set False to save space)

# Performance optimization
ENABLE_BATCH_PROCESSING = True  # Process multiple frames at once when possible
ENABLE_PARALLEL_WINDOWS = False  # Process windows in parallel (experimental)
BATCH_SIZE = 4  # Number of frames to process together

# Weapon/risk keywords (using actual YOLO COCO classes)
# YOLO COCO classes: 43=knife, 34=baseball bat, 76=scissors
WEAPON_YOLO_CLASSES = ["knife", "baseball bat", "scissors"]  # Actual YOLO class names
WEAPON_KEYWORDS = ["gun", "weapon", "pistol", "rifle", "sword"]  # For caption/tag matching
VIOLENT_CAPTION_KEYWORDS = ["fight", "attack", "hit", "violence", "strike", "punch", "kick", "beat", "hurt", "blood", "injured", "wound"]

# Context-aware keywords (safe contexts)
SAFE_CONTEXT_KEYWORDS = ["kitchen", "cooking", "chef", "restaurant", "cutting board", "food", "vegetable", "tomato", "pepper"]
SAFE_SCENE_TAGS = ["kitchen", "restaurant", "home", "cooking", "food preparation"]
