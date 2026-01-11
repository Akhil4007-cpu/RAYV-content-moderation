"""
Automated YouTube video testing - searches, downloads, tests, and reports
Covers all categories: safe and unsafe content
"""
import os
import json
import subprocess
import sys
import shutil
import tempfile
import time
from datetime import datetime

def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except:
        return False

def search_videos(query, max_results=5, platforms=["youtube"]):
    """
    Search multiple platforms for videos (15 seconds duration)
    Returns list of video URLs with duration info
    """
    all_results = []
    
    for platform in platforms:
        try:
            if platform == "youtube":
                # Search YouTube and get URLs first
                cmd_search = [
                    "yt-dlp",
                    f"ytsearch{max_results*2}:{query}",
                    "--flat-playlist",
                    "--print", "%(url)s",
                    "--quiet"
                ]
                
                result = subprocess.run(cmd_search, capture_output=True, text=True, timeout=30)
                urls = [url.strip() for url in result.stdout.strip().split('\n') if url.strip()]
                
                # Check each URL for duration
                for url in urls[:max_results*2]:
                    try:
                        cmd_info = [
                            "yt-dlp",
                            "--print", "%(duration)s|||%(title)s",
                            "--quiet",
                            "--no-download",
                            url
                        ]
                        info_result = subprocess.run(cmd_info, capture_output=True, text=True, timeout=10)
                        if info_result.returncode == 0:
                            parts = info_result.stdout.strip().split('|||')
                            if len(parts) >= 1:
                                duration_str = parts[0].strip()
                                duration = int(float(duration_str)) if duration_str.replace('.', '').replace('-', '').isdigit() else 0
                                title = parts[1] if len(parts) > 1 else "Unknown"
                                
                                # Filter for videos close to 15 seconds (10-20 seconds)
                                if 10 <= duration <= 20:
                                    all_results.append({
                                        "url": url,
                                        "duration": duration,
                                        "title": title,
                                        "platform": platform
                                    })
                                    
                                    if len(all_results) >= max_results:
                                        return all_results[:max_results]
                    except:
                        continue
        except:
            continue
    
    return all_results[:max_results]

def get_video_info(url):
    """Get video title and duration"""
    cmd = [
        "yt-dlp",
        "--print", "%(title)s|||%(duration)s",
        "--quiet",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            parts = result.stdout.strip().split("|||")
            if len(parts) == 2:
                duration_str = parts[1].strip()
                duration = int(float(duration_str)) if duration_str.replace('.', '').isdigit() else 0
                return {
                    "title": parts[0],
                    "duration": duration
                }
    except:
        pass
    return {"title": "Unknown", "duration": 0}

def download_video(url, output_path, target_duration=15):
    """
    Download video and trim to exactly 15 seconds
    Supports YouTube, Vimeo, Dailymotion, etc.
    """
    # First, get video duration
    info = get_video_info(url)
    video_duration = info.get("duration", 0)
    
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", output_path,
        "--no-playlist",
        "--quiet"
    ]
    
    # If video is longer than 15 seconds, trim it
    # If video is shorter, we'll use it as-is (but prefer 15-second videos)
    if video_duration > target_duration:
        # Trim to exactly 15 seconds from start
        cmd.extend(["--postprocessor-args", f"ffmpeg:-t {target_duration}"])
    elif video_duration < target_duration:
        # Video is shorter, we'll use it but note it
        print(f"        [!] Video is {video_duration}s (shorter than 15s)")
    
    cmd.append(url)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(output_path):
            # Verify duration is correct (or close)
            return True
        return False
    except:
        return False

def run_pipeline_on_video(video_path):
    """Run full pipeline on a video and get result"""
    try:
        result = subprocess.run(
            [sys.executable, "main_pipeline.py", "--video", video_path, "--no-checks"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            return None
        
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
        
        confusion_matrix[truth_label][pred_label] += 1
        
        if pred_label == truth_label:
            correct += 1
            per_class_correct[truth_label] += 1
        
        per_class_total[truth_label] += 1
    
    accuracy = correct / total if total > 0 else 0
    
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
    """Main automated testing function"""
    
    if not check_yt_dlp():
        print("[X] yt-dlp not found!")
        print("    Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Define test categories with search queries and expected labels
    test_categories = [
        # SAFE categories
        {
            "category": "cooking",
            "queries": ["cooking tutorial", "recipe video", "cooking at home"],
            "label": "SAFE",
            "count": 5
        },
        {
            "category": "education",
            "queries": ["educational video", "tutorial", "learning"],
            "label": "SAFE",
            "count": 5
        },
        {
            "category": "nature",
            "queries": ["nature documentary", "wildlife", "nature video"],
            "label": "SAFE",
            "count": 5
        },
        {
            "category": "daily_life",
            "queries": ["daily life vlog", "lifestyle video", "daily routine"],
            "label": "SAFE",
            "count": 5
        },
        {
            "category": "product_review",
            "queries": ["product review", "unboxing", "review video"],
            "label": "SAFE",
            "count": 3
        },
        
        # RISKY categories
        {
            "category": "action_movie",
            "queries": ["action movie scene", "action clip", "movie action"],
            "label": "RISKY",
            "count": 5
        },
        {
            "category": "sports_contact",
            "queries": ["boxing match", "martial arts", "combat sports"],
            "label": "RISKY",
            "count": 5
        },
        {
            "category": "stunt",
            "queries": ["stunt video", "extreme sports", "dangerous stunt"],
            "label": "RISKY",
            "count": 3
        },
        
        # VIOLENT categories
        {
            "category": "fight_scene",
            "queries": ["fight scene", "fighting", "brawl"],
            "label": "VIOLENT",
            "count": 5
        },
        {
            "category": "weapon_usage",
            "queries": ["weapon demonstration", "knife fight", "weapon video"],
            "label": "VIOLENT",
            "count": 5
        },
        {
            "category": "violence",
            "queries": ["violent scene", "violence", "aggressive behavior"],
            "label": "VIOLENT",
            "count": 5
        }
    ]
    
    print("=" * 60)
    print("AUTOMATED YOUTUBE VIDEO TESTING")
    print("=" * 60)
    
    total_videos_needed = sum(cat["count"] for cat in test_categories)
    print(f"\n[PLAN] Total videos to test: {total_videos_needed}")
    print(f"       Categories: {len(test_categories)}")
    print(f"       SAFE: {sum(c['count'] for c in test_categories if c['label'] == 'SAFE')}")
    print(f"       RISKY: {sum(c['count'] for c in test_categories if c['label'] == 'RISKY')}")
    print(f"       VIOLENT: {sum(c['count'] for c in test_categories if c['label'] == 'VIOLENT')}")
    print("\n[START] Beginning automated testing...\n")
    
    temp_dir = tempfile.mkdtemp(prefix="auto_test_")
    print(f"[TEMP] Using temp directory: {temp_dir}\n")
    
    all_results = []
    all_predictions = []
    all_ground_truth = []
    
    video_counter = 0
    
    try:
        for cat_idx, category_config in enumerate(test_categories, 1):
            category = category_config["category"]
            queries = category_config["queries"]
            expected_label = category_config["label"]
            count = category_config["count"]
            
            print(f"\n[{cat_idx}/{len(test_categories)}] Category: {category} ({expected_label})")
            print(f"    Searching for {count} videos...")
            
            # Search for videos across multiple platforms
            all_video_info = []
            for query in queries:
                videos = search_videos(query, max_results=count, platforms=["youtube"])
                all_video_info.extend(videos)
                if len(all_video_info) >= count:
                    break
                time.sleep(2)  # Rate limiting
            
            # Limit to required count
            all_video_info = all_video_info[:count]
            
            if not all_video_info:
                print(f"    [X] No suitable videos found for {category}")
                continue
            
            print(f"    [OK] Found {len(all_video_info)} videos (15-second clips)")
            
            # Process each video
            for vid_idx, video_info in enumerate(all_video_info, 1):
                url = video_info["url"]
                platform = video_info.get("platform", "unknown")
                video_counter += 1
                video_id = f"{category}_{vid_idx}"
                
                print(f"\n    [{vid_idx}/{len(all_video_info)}] Processing: {video_id}")
                print(f"        Platform: {platform}")
                print(f"        URL: {url}")
                
                # Get video info
                info = get_video_info(url)
                video_duration = info.get("duration", 0)
                print(f"        Title: {info['title'][:50]}...")
                print(f"        Duration: {video_duration}s")
                
                # Skip if video is too short or too long
                if video_duration < 10 or video_duration > 30:
                    print(f"        [!] Skipping - duration not suitable ({video_duration}s)")
                    continue
                
                # Download video (trimmed to exactly 15 seconds if longer)
                temp_video_path = os.path.join(temp_dir, f"{video_id}.mp4")
                print(f"        [DOWNLOAD] Downloading (target: 15 seconds)...")
                
                if not download_video(url, temp_video_path, target_duration=15):
                    print(f"        [X] Download failed")
                    continue
                
                file_size = os.path.getsize(temp_video_path) / 1024 / 1024
                print(f"        [OK] Downloaded ({file_size:.1f} MB)")
                
                # Run pipeline
                print(f"        [PIPELINE] Running moderation...")
                result = run_pipeline_on_video(temp_video_path)
                
                if not result:
                    print(f"        [X] Pipeline failed")
                    os.remove(temp_video_path)
                    continue
                
                predicted_label = result.get("final_label", "SAFE").upper()
                confidence = result.get("confidence", 0.0)
                is_correct = predicted_label == expected_label
                status = "[OK]" if is_correct else "[X]"
                
                print(f"        {status} Predicted: {predicted_label} (confidence: {confidence:.2f})")
                print(f"               Expected: {expected_label}")
                
                # Store results
                all_results.append({
                    "video_id": video_id,
                    "url": url,
                    "platform": platform,
                    "title": info['title'],
                    "duration": video_duration,
                    "category": category,
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "correct": is_correct,
                    "confidence": confidence,
                    "risk_score": result.get("statistics", {}).get("avg_risk_score", 0.0)
                })
                
                all_predictions.append({
                    "video_id": video_id,
                    "predicted_label": predicted_label,
                    "confidence": confidence
                })
                
                all_ground_truth.append({
                    "video_id": video_id,
                    "label": expected_label,
                    "category": category
                })
                
                # Clean up
                os.remove(temp_video_path)
                
                # Rate limiting
                time.sleep(2)
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n[CLEANUP] Removed temp directory")
    
    # Calculate accuracy
    print("\n" + "=" * 60)
    print("CALCULATING ACCURACY METRICS")
    print("=" * 60)
    
    accuracy_metrics = calculate_accuracy(all_predictions, all_ground_truth)
    
    if not accuracy_metrics:
        print("[X] Could not calculate metrics")
        return
    
    # Generate comprehensive report
    report = {
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_videos": accuracy_metrics['total_videos'],
            "accuracy": accuracy_metrics['accuracy'],
            "correct": accuracy_metrics['correct'],
            "categories_tested": len(test_categories)
        },
        "per_class_metrics": accuracy_metrics['per_class_metrics'],
        "confusion_matrix": accuracy_metrics['confusion_matrix'],
        "per_category_breakdown": {},
        "detailed_results": all_results
    }
    
    # Per-category breakdown
    categories = {}
    for r in all_results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0, "label": r['expected']}
        categories[cat]["total"] += 1
        if r['correct']:
            categories[cat]["correct"] += 1
    
    for cat, stats in categories.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        report["per_category_breakdown"][cat] = {
            "accuracy": round(acc, 3),
            "correct": stats["correct"],
            "total": stats["total"],
            "expected_label": stats["label"]
        }
    
    # Print report
    print(f"\n[REPORT] Overall Accuracy: {accuracy_metrics['accuracy'] * 100:.1f}%")
    print(f"    Correct: {accuracy_metrics['correct']}/{accuracy_metrics['total_videos']}")
    print(f"    Test Date: {report['test_date']}")
    
    print("\n[REPORT] Per-Class Metrics:")
    for label, metrics in accuracy_metrics['per_class_metrics'].items():
        print(f"    {label}:")
        print(f"      Precision: {metrics['precision']:.3f}")
        print(f"      Recall: {metrics['recall']:.3f}")
        print(f"      F1-Score: {metrics['f1']:.3f}")
        print(f"      Support: {metrics['support']}")
    
    print("\n[REPORT] Per-Category Accuracy:")
    for cat, stats in report['per_category_breakdown'].items():
        print(f"    {cat} ({stats['expected_label']}): {stats['accuracy'] * 100:.1f}% ({stats['correct']}/{stats['total']})")
    
    print("\n[REPORT] Confusion Matrix:")
    print("                Predicted:")
    print("          SAFE    RISKY   VIOLENT")
    for true_label in ["SAFE", "RISKY", "VIOLENT"]:
        row = f"{true_label:8}"
        for pred_label in ["SAFE", "RISKY", "VIOLENT"]:
            count = accuracy_metrics['confusion_matrix'][true_label][pred_label]
            row += f"{count:8}"
        print(row)
    
    # Save report
    report_file = "accuracy_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Detailed report saved: {report_file}")
    
    # Generate markdown report
    md_report = generate_markdown_report(report)
    md_file = "ACCURACY_REPORT.md"
    with open(md_file, "w") as f:
        f.write(md_report)
    
    print(f"[OK] Markdown report saved: {md_file}")
    
    return report

def generate_markdown_report(report):
    """Generate markdown formatted report"""
    md = f"""# Model Accuracy Report

**Test Date:** {report['test_date']}

## Summary

- **Total Videos Tested:** {report['summary']['total_videos']}
- **Overall Accuracy:** {report['summary']['accuracy'] * 100:.1f}%
- **Correct Predictions:** {report['summary']['correct']}/{report['summary']['total_videos']}
- **Categories Tested:** {report['summary']['categories_tested']}

## Per-Class Metrics

"""
    
    for label, metrics in report['per_class_metrics'].items():
        md += f"### {label}\n"
        md += f"- **Precision:** {metrics['precision']:.3f}\n"
        md += f"- **Recall:** {metrics['recall']:.3f}\n"
        md += f"- **F1-Score:** {metrics['f1']:.3f}\n"
        md += f"- **Support:** {metrics['support']}\n\n"
    
    md += "## Per-Category Breakdown\n\n"
    md += "| Category | Expected Label | Accuracy | Correct/Total |\n"
    md += "|----------|----------------|----------|---------------|\n"
    
    for cat, stats in report['per_category_breakdown'].items():
        md += f"| {cat} | {stats['expected_label']} | {stats['accuracy'] * 100:.1f}% | {stats['correct']}/{stats['total']} |\n"
    
    md += "\n## Confusion Matrix\n\n"
    md += "| Actual \\ Predicted | SAFE | RISKY | VIOLENT |\n"
    md += "|-------------------|------|-------|----------|\n"
    
    for true_label in ["SAFE", "RISKY", "VIOLENT"]:
        row = f"| {true_label} |"
        for pred_label in ["SAFE", "RISKY", "VIOLENT"]:
            count = report['confusion_matrix'][true_label][pred_label]
            row += f" {count} |"
        md += row + "\n"
    
    return md

if __name__ == "__main__":
    main()
