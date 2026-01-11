"""
Test model accuracy on YouTube videos
Downloads videos temporarily, runs pipeline, generates accuracy report
"""
import os
import json
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_youtube_video(url, output_path, duration=15):
    """Download YouTube video (first 15 seconds)"""
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        "--postprocessor-args", f"ffmpeg:-t {duration}",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except:
        return False

def run_pipeline_on_video(video_path):
    """Run full pipeline on a video and get result"""
    try:
        # Run pipeline
        result = subprocess.run(
            [sys.executable, "main_pipeline.py", "--video", video_path, "--no-checks"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max per video
        )
        
        # Find the result file in outputs folder
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            return None
        
        # Find latest session folder
        sessions = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
        if not sessions:
            return None
        
        latest_session = max(sessions, key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)))
        result_file = os.path.join(outputs_dir, latest_session, "final_moderation_result.json")
        
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"[X] Error processing video: {e}")
        return None

def calculate_accuracy(predictions, ground_truth):
    """Calculate accuracy metrics"""
    if not predictions or not ground_truth:
        return None
    
    total = len(predictions)
    correct = 0
    confusion_matrix = {
        "SAFE": {"SAFE": 0, "RISKY": 0, "VIOLENT": 0},
        "RISKY": {"SAFE": 0, "RISKY": 0, "VIOLENT": 0},
        "VIOLENT": {"SAFE": 0, "RISKY": 0, "VIOLENT": 0}
    }
    
    per_class_correct = {"SAFE": 0, "RISKY": 0, "VIOLENT": 0}
    per_class_total = {"SAFE": 0, "RISKY": 0, "VIOLENT": 0}
    
    for pred, truth in zip(predictions, ground_truth):
        pred_label = pred.get("predicted_label", "SAFE")
        truth_label = truth.get("label", "SAFE")
        
        # Update confusion matrix
        confusion_matrix[truth_label][pred_label] += 1
        
        # Count correct
        if pred_label == truth_label:
            correct += 1
            per_class_correct[truth_label] += 1
        
        per_class_total[truth_label] += 1
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Per-class precision, recall, F1
    metrics = {}
    for label in ["SAFE", "RISKY", "VIOLENT"]:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in ["SAFE", "RISKY", "VIOLENT"] if other != label)
        fn = sum(confusion_matrix[label][other] for other in ["SAFE", "RISKY", "VIOLENT"] if other != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": per_class_total[label]
        }
    
    return {
        "accuracy": round(accuracy, 3),
        "total_videos": total,
        "correct": correct,
        "confusion_matrix": confusion_matrix,
        "per_class_metrics": metrics
    }

def main():
    """Main testing function"""
    
    # Check yt-dlp
    if not check_yt_dlp():
        print("[X] yt-dlp not found!")
        print("    Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Load test configuration
    config_file = "test_videos_config.json"
    if not os.path.exists(config_file):
        print(f"[!] {config_file} not found. Creating template...")
        create_template_config(config_file)
        print(f"[OK] Template created. Edit {config_file} and add YouTube URLs with ground truth labels.")
        return
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    videos = config.get("videos", [])
    if not videos:
        print("[!] No videos in configuration")
        return
    
    print("=" * 60)
    print("YOUTUBE VIDEO ACCURACY TEST")
    print("=" * 60)
    print(f"Testing {len(videos)} videos...\n")
    
    # Create temp directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="youtube_test_")
    print(f"[TEMP] Using temp directory: {temp_dir}")
    
    predictions = []
    ground_truth = []
    results = []
    
    try:
        for i, video_config in enumerate(videos, 1):
            video_id = video_config.get("id", f"video_{i}")
            url = video_config.get("url", "")
            expected_label = video_config.get("label", "SAFE").upper()
            category = video_config.get("category", "general")
            
            if not url:
                print(f"[!] Skipping {video_id}: No URL")
                continue
            
            print(f"\n[{i}/{len(videos)}] Processing: {video_id}")
            print(f"    URL: {url}")
            print(f"    Expected: {expected_label} | Category: {category}")
            
            # Download video temporarily
            temp_video_path = os.path.join(temp_dir, f"{video_id}.mp4")
            print(f"    [DOWNLOAD] Downloading...")
            
            if not download_youtube_video(url, temp_video_path):
                print(f"    [X] Failed to download")
                continue
            
            print(f"    [OK] Downloaded ({os.path.getsize(temp_video_path) / 1024 / 1024:.1f} MB)")
            
            # Run pipeline
            print(f"    [PIPELINE] Running moderation pipeline...")
            result = run_pipeline_on_video(temp_video_path)
            
            if not result:
                print(f"    [X] Pipeline failed")
                continue
            
            predicted_label = result.get("final_label", "SAFE").upper()
            confidence = result.get("confidence", 0.0)
            
            # Check if correct
            is_correct = predicted_label == expected_label
            status = "[OK]" if is_correct else "[X]"
            
            print(f"    {status} Predicted: {predicted_label} (confidence: {confidence:.2f})")
            print(f"       Expected: {expected_label}")
            
            # Store results
            predictions.append({
                "video_id": video_id,
                "url": url,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "risk_score": result.get("statistics", {}).get("avg_risk_score", 0.0)
            })
            
            ground_truth.append({
                "video_id": video_id,
                "label": expected_label,
                "category": category
            })
            
            results.append({
                "video_id": video_id,
                "url": url,
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": is_correct,
                "confidence": confidence,
                "category": category
            })
            
            # Clean up temp video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n[CLEANUP] Removed temp directory")
    
    # Calculate accuracy
    print("\n" + "=" * 60)
    print("CALCULATING ACCURACY METRICS")
    print("=" * 60)
    
    accuracy_metrics = calculate_accuracy(predictions, ground_truth)
    
    if not accuracy_metrics:
        print("[X] Could not calculate metrics")
        return
    
    # Print report
    print(f"\n[REPORT] Overall Accuracy: {accuracy_metrics['accuracy'] * 100:.1f}%")
    print(f"    Correct: {accuracy_metrics['correct']}/{accuracy_metrics['total_videos']}")
    
    print("\n[REPORT] Per-Class Metrics:")
    for label, metrics in accuracy_metrics['per_class_metrics'].items():
        print(f"    {label}:")
        print(f"      Precision: {metrics['precision']:.3f}")
        print(f"      Recall: {metrics['recall']:.3f}")
        print(f"      F1-Score: {metrics['f1']:.3f}")
        print(f"      Support: {metrics['support']}")
    
    print("\n[REPORT] Confusion Matrix:")
    print("                Predicted:")
    print("          SAFE    RISKY   VIOLENT")
    for true_label in ["SAFE", "RISKY", "VIOLENT"]:
        row = f"{true_label:8}"
        for pred_label in ["SAFE", "RISKY", "VIOLENT"]:
            count = accuracy_metrics['confusion_matrix'][true_label][pred_label]
            row += f"{count:8}"
        print(row)
    
    # Save detailed report
    report = {
        "summary": {
            "total_videos": accuracy_metrics['total_videos'],
            "accuracy": accuracy_metrics['accuracy'],
            "correct": accuracy_metrics['correct']
        },
        "per_class_metrics": accuracy_metrics['per_class_metrics'],
        "confusion_matrix": accuracy_metrics['confusion_matrix'],
        "detailed_results": results
    }
    
    report_file = "accuracy_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Detailed report saved: {report_file}")
    
    # Print per-category breakdown
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r['correct']:
            categories[cat]["correct"] += 1
    
    if categories:
        print("\n[REPORT] Per-Category Accuracy:")
        for cat, stats in categories.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"    {cat}: {acc * 100:.1f}% ({stats['correct']}/{stats['total']})")

def create_template_config(filename):
    """Create template test configuration"""
    template = {
        "videos": [
            {
                "id": "safe_cooking_1",
                "url": "https://www.youtube.com/watch?v=EXAMPLE1",
                "label": "SAFE",
                "category": "cooking",
                "description": "Cooking video - should be safe"
            },
            {
                "id": "risky_action_1",
                "url": "https://www.youtube.com/watch?v=EXAMPLE2",
                "label": "RISKY",
                "category": "action",
                "description": "Action scene - might be risky"
            },
            {
                "id": "violent_fight_1",
                "url": "https://www.youtube.com/watch?v=EXAMPLE3",
                "label": "VIOLENT",
                "category": "violence",
                "description": "Fight scene - should be violent"
            }
        ]
    }
    
    with open(filename, "w") as f:
        json.dump(template, f, indent=2)

if __name__ == "__main__":
    main()
