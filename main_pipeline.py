"""
Main Pipeline: Vision AI Video Moderation
Unified entry point for the complete video moderation pipeline

Pipeline Steps:
1. Extract frames from video (step7a)
2. Remove duplicate frames (step7b)
3. Build temporal windows (step7c)
4. Analyze windows with multimodal models (step7d)
5. Make final moderation decision (step8)

Optional individual model tests:
- Step 2: YOLO object detection
- Step 3: RAM scene tagging
- Step 5: BLIP image captioning
"""
import os
import sys
import argparse
import subprocess
import json
import shutil
import datetime
from pathlib import Path

import config

def run_step(script_name, description):
    """Run a pipeline step and handle errors"""
    return run_step_with_args(script_name, description, [])

def run_step_with_args(script_name, description, extra_args=[]):
    """Run a pipeline step with additional arguments"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"[X] Script not found: {script_name}")
        return False
    
    try:
        cmd = [sys.executable, script_name] + extra_args
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] Error in {description}: {e}")
        return False
    except Exception as e:
        print(f"[X] Unexpected error in {description}: {e}")
        return False

def test_individual_model(model_name, script_name, image_path="test.jpg"):
    """Test individual models for debugging"""
    print(f"\n{'='*60}")
    print(f"[TEST] Testing {model_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"[X] Script not found: {script_name}")
        return False
    
    if model_name in ["YOLO", "RAM"] and not os.path.exists(image_path):
        print(f"[!] Warning: Test image '{image_path}' not found, skipping {model_name} test")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"[OK] {model_name} test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] Error testing {model_name}: {e}")
        return False
    except Exception as e:
        print(f"[X] Unexpected error testing {model_name}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("[CHECK] Checking dependencies...")
    required_packages = [
        "torch", "torchvision", "ultralytics", "transformers",
        "PIL", "cv2", "numpy"
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == "PIL":
                __import__("PIL")
            elif package == "cv2":
                __import__("cv2")
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[X] Missing dependencies: {', '.join(missing)}")
        print("   Please install requirements: pip install -r requirements.txt")
        return False
    
    print("[OK] All dependencies installed")
    return True

def check_model_files():
    """Check if model checkpoint files exist"""
    print("[CHECK] Checking model files...")
    
    issues = []
    
    # Check YOLO model (will auto-download if missing)
    if not os.path.exists(config.YOLO_MODEL_PATH):
        print(f"[!] YOLO model will be auto-downloaded: {config.YOLO_MODEL_PATH}")
    
    # Check RAM checkpoint
    if not os.path.exists(config.RAM_CHECKPOINT_PATH):
        issues.append(f"RAM checkpoint not found: {config.RAM_CHECKPOINT_PATH}")
        print(f"   Please download from: https://huggingface.co/spaces/xinyu1205/recognize-anything")
    
    if issues:
        print("[!] Model file warnings (processing may fail):")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("[OK] Model files present")
    return True

def create_session_dir(video_path):
    """Create a session directory based on video name and timestamp"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Clean filename for directory name (remove invalid characters)
    video_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in video_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{video_name}_{timestamp}"
    session_dir = os.path.join(config.OUTPUT_BASE_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir, session_name

def clean_previous_results(clean_all=False):
    """Clean previous processing results"""
    dirs_to_clean = [config.FRAMES_DIR_RAW, config.FRAMES_DIR_UNIQUE]
    files_to_clean = [
        config.WINDOW_RESULTS_FILE,
        "final_moderation_result.json",
        "windows_info.json"
    ]
    
    cleaned = []
    
    # Clean directories
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            if clean_all:
                shutil.rmtree(dir_path)
                cleaned.append(f"Directory: {dir_path}")
            else:
                # Only clean contents, not directory itself
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        cleaned.append(f"File: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        cleaned.append(f"Directory: {item_path}")
    
    # Clean files
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            cleaned.append(f"File: {file_path}")
    
    if cleaned:
        print(f"[CLEAN] Cleaned {len(cleaned)} items:")
        for item in cleaned[:5]:  # Show first 5
            print(f"   - {item}")
        if len(cleaned) > 5:
            print(f"   ... and {len(cleaned) - 5} more")
    else:
        print("[OK] No previous results to clean")
    
    return len(cleaned)

def archive_previous_results(session_dir):
    """Archive previous results to session directory"""
    items_to_archive = {
        config.FRAMES_DIR_RAW: "frames_raw",
        config.FRAMES_DIR_UNIQUE: "frames_unique",
        config.WINDOW_RESULTS_FILE: "window_results.json",
        "final_moderation_result.json": "final_moderation_result.json",
        "windows_info.json": "windows_info.json"
    }
    
    archived = []
    archive_dir = os.path.join(session_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    
    for source, dest_name in items_to_archive.items():
        if os.path.exists(source):
            dest = os.path.join(archive_dir, dest_name)
            if os.path.isdir(source):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
                archived.append(f"Directory: {source} -> {dest}")
            else:
                shutil.copy2(source, dest)
                archived.append(f"File: {source} -> {dest}")
    
    return archived

def main():
    parser = argparse.ArgumentParser(
        description="Vision AI Video Moderation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full video moderation pipeline
  python main_pipeline.py --video input_video.mp4
  
  # Test individual models
  python main_pipeline.py --test-models
  
  # Run specific steps only
  python main_pipeline.py --step extract --step remove_duplicates
  
  # Skip dependency checks
  python main_pipeline.py --video input_video.mp4 --no-checks
  
  # Clean previous results before processing
  python main_pipeline.py --video input_video.mp4 --clean
  
  # Keep previous results in archive
  python main_pipeline.py --video input_video.mp4 --keep-results
  
  # Use custom output directory
  python main_pipeline.py --video input_video.mp4 --output-dir outputs/my_session
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video file (default: input_video.mp4)"
    )
    
    parser.add_argument(
        "--test-models",
        action="store_true",
        help="Test individual models (YOLO, RAM, BLIP) instead of running pipeline"
    )
    
    parser.add_argument(
        "--step",
        action="append",
        choices=["extract", "remove_duplicates", "build_windows", "analyze", "decision", "all"],
        help="Run specific step(s) only (can be used multiple times)"
    )
    
    parser.add_argument(
        "--no-checks",
        action="store_true",
        help="Skip dependency and file checks"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file (default: config.py)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previous results before processing"
    )
    
    parser.add_argument(
        "--keep-results",
        action="store_true",
        help="Keep previous results in output archive (default: clean them)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory (default: outputs/video_name_timestamp)"
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        if os.path.exists(args.config):
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_config", args.config)
            custom_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config)
            # Merge with default config
            for key in dir(custom_config):
                if not key.startswith("_"):
                    setattr(config, key, getattr(custom_config, key))
        else:
            print(f"[X] Config file not found: {args.config}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("Vision AI Video Moderation Pipeline")
    print("="*60)
    
    # Run checks
    if not args.no_checks:
        if not check_dependencies():
            sys.exit(1)
        check_model_files()  # Warning only, don't exit
    
    # Test models mode
    if args.test_models:
        print("\n[TEST] Running individual model tests...")
        test_individual_model("YOLO", "step2_yolo_eyes.py")
        test_individual_model("RAM", "step3_ram_eyes.py")
        test_individual_model("BLIP", "step5_brain_reasoning.py")
        print("\n[OK] Model testing complete")
        return
    
    # Determine steps to run
    if args.step:
        steps_to_run = args.step
        if "all" in steps_to_run:
            steps_to_run = ["extract", "remove_duplicates", "build_windows", "analyze", "decision"]
    else:
        steps_to_run = ["extract", "remove_duplicates", "build_windows", "analyze", "decision"]
    
    # Check video file
    video_path = args.video if args.video else "input_video.mp4"
    session_dir = None
    session_name = None
    
    # Create session directory for outputs
    if "extract" in steps_to_run:
        if not os.path.exists(video_path):
            print(f"\n[X] Video file not found: {video_path}")
            print("   Please provide a valid video file using --video or place 'input_video.mp4' in the project directory")
            sys.exit(1)
        
        # Create session directory
        if args.output_dir:
            session_dir = args.output_dir
            session_name = os.path.basename(session_dir)
        else:
            session_dir, session_name = create_session_dir(video_path)
        
        os.makedirs(session_dir, exist_ok=True)
        
        # Update config with session paths
        config.FRAMES_DIR_RAW = os.path.join(session_dir, "frames_raw")
        config.FRAMES_DIR_UNIQUE = os.path.join(session_dir, "frames_unique")
        config.WINDOW_RESULTS_FILE = os.path.join(session_dir, "window_results.json")
        
        print(f"\n[DIR] Session directory: {session_dir}")
        
        # ALWAYS clean outputs folder before new run (unless --keep-results is specified)
        if not args.keep_results:
            print("\n[CLEAN] Cleaning outputs folder for new run...")
            # Clean entire outputs folder except current session
            if os.path.exists(config.OUTPUT_BASE_DIR):
                for item in os.listdir(config.OUTPUT_BASE_DIR):
                    item_path = os.path.join(config.OUTPUT_BASE_DIR, item)
                    if item_path != session_dir:  # Don't delete current session
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            print(f"   Removed: {item_path}")
                        elif os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"   Removed: {item_path}")
            
            # Clean old root-level result files if they exist
            old_result_files = [
                "window_results.json",
                "final_moderation_result.json",
                "windows_info.json"
            ]
            for old_file in old_result_files:
                if os.path.exists(old_file):
                    os.remove(old_file)
                    print(f"   Removed: {old_file}")
            
            # Clean old frame directories if they exist at root
            old_frames_raw = "frames_raw"
            old_frames_unique = "frames_unique"
            if os.path.exists(old_frames_raw):
                shutil.rmtree(old_frames_raw)
                print(f"   Removed: {old_frames_raw}")
            if os.path.exists(old_frames_unique):
                shutil.rmtree(old_frames_unique)
                print(f"   Removed: {old_frames_unique}")
            
            print("[OK] Outputs folder cleaned - ready for new run")
        else:
            print("\n[KEEP] Keeping previous results (--keep-results flag set)")
    
    # Set output paths for results
    final_result_file = os.path.join(session_dir, "final_moderation_result.json") if session_dir else "final_moderation_result.json"
    
    # Run pipeline steps
    success = True
    
    step_map = {
        "extract": ("step7a_extract_frames.py", "Step 7a: Extract Frames from Video", ["--video", video_path]),
        "remove_duplicates": ("step7b_remove_duplicates.py", "Step 7b: Remove Duplicate Frames", []),
        "build_windows": ("step7c_build_windows.py", "Step 7c: Build Temporal Windows", []),
        "analyze": ("step7d_window_multimodal_analysis.py", "Step 7d: Window Multimodal Analysis", []),
        "decision": ("step8_video_decision_engine.py", "Step 8: Video Decision Engine", [])
    }
    
    for step in steps_to_run:
        if step in step_map:
            script, description, extra_args = step_map[step]
            
            # Update run_step to accept extra arguments
            if not run_step_with_args(script, description, extra_args):
                success = False
                print(f"\n[X] Pipeline failed at {description}")
                print("   Fix the error above and re-run, or use --step to continue from next step")
                break
        else:
            print(f"[!] Unknown step: {step}")
    
    if success:
        print("\n" + "="*60)
        print("[OK] PIPELINE COMPLETE")
        print("="*60)
        
        # Display final result if decision step was run
        result_file = final_result_file if session_dir else "final_moderation_result.json"
        if "decision" in steps_to_run and os.path.exists(result_file):
            print("\n[RESULT] Final Moderation Result:")
            with open(result_file, "r") as f:
                result = json.load(f)
            print(f"   Label: {result['final_label']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Reason: {result['reason']}")
            
            if session_dir:
                print(f"\n[OUTPUT] All results saved to: {session_dir}")
                print(f"   - Final result: {result_file}")
                print(f"   - Window analysis: {config.WINDOW_RESULTS_FILE}")
                print(f"   - Window info: {os.path.join(session_dir, 'windows_info.json')}")
                print(f"   - Raw frames: {config.FRAMES_DIR_RAW}")
                print(f"   - Unique frames: {config.FRAMES_DIR_UNIQUE}")
            else:
                print(f"\n   Full result saved to: {result_file}")
    else:
        print("\n" + "="*60)
        print("[X] PIPELINE FAILED")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
