"""
Simple Model Testing - Test on local videos or provide YouTube URLs
"""
import os
import json
import subprocess
import sys
from pathlib import Path

def run_pipeline(video_path):
    """Run pipeline on a video and return result"""
    try:
        result = subprocess.run(
            [sys.executable, "main_pipeline.py", "--video", video_path, "--no-checks"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Find result file
        outputs_dir = "outputs"
        if os.path.exists(outputs_dir):
            sessions = [d for d in os.listdir(outputs_dir) 
                      if os.path.isdir(os.path.join(outputs_dir, d))]
            if sessions:
                latest = max(sessions, key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)))
                result_file = os.path.join(outputs_dir, latest, "final_moderation_result.json")
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        return json.load(f)
        return None
    except Exception as e:
        print(f"[X] Error: {e}")
        return None

def test_single_video(video_path, expected_label=None):
    """Test a single video"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(video_path):
        print(f"[X] Video not found: {video_path}")
        return None
    
    print("[PIPELINE] Running moderation...")
    result = run_pipeline(video_path)
    
    if not result:
        print("[X] Pipeline failed")
        return None
    
    predicted = result.get("final_label", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    
    print(f"[RESULT] Predicted: {predicted}")
    print(f"         Confidence: {confidence:.2f}")
    print(f"         Risk Score: {result.get('statistics', {}).get('avg_risk_score', 0):.2f}")
    
    if expected_label:
        is_correct = predicted.upper() == expected_label.upper()
        status = "[OK]" if is_correct else "[X]"
        print(f"{status} Expected: {expected_label}")
        return {"predicted": predicted, "expected": expected_label, "correct": is_correct}
    
    return {"predicted": predicted, "confidence": confidence}

def main():
    """Main function"""
    print("=" * 60)
    print("MODEL TESTING")
    print("=" * 60)
    
    # Check for test videos directory
    test_dir = "test_videos"
    if not os.path.exists(test_dir):
        print(f"\n[!] No test_videos directory found")
        print(f"    Create '{test_dir}' folder and add your test videos")
        print(f"    Or provide video path as argument:")
        print(f"    python test_model.py path/to/video.mp4")
        return
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    videos = []
    for ext in video_extensions:
        videos.extend(Path(test_dir).glob(f"*{ext}"))
        videos.extend(Path(test_dir).glob(f"*{ext.upper()}"))
    
    if not videos:
        print(f"\n[!] No videos found in {test_dir}/")
        print(f"    Add .mp4, .avi, .mov files to test")
        return
    
    print(f"\n[FOUND] {len(videos)} video(s) to test")
    
    results = []
    for i, video_path in enumerate(videos, 1):
        result = test_single_video(str(video_path))
        if result:
            results.append({
                "video": os.path.basename(video_path),
                **result
            })
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['video']}: {r['predicted']}")
        
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to test_results.json")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test single video provided as argument
        video_path = sys.argv[1]
        expected = sys.argv[2] if len(sys.argv) > 2 else None
        test_single_video(video_path, expected)
    else:
        # Test all videos in test_videos directory
        main()
