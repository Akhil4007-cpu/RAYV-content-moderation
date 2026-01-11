"""
Step 8: Video Decision Engine
Aggregates window analysis results and makes final moderation decision
"""
import json
import os
import sys
import numpy as np
import config

# ===============================
# CONFIG
# ===============================
WINDOW_RESULTS_FILE = config.WINDOW_RESULTS_FILE
ESCALATION_THRESHOLD = config.ESCALATION_THRESHOLD
VIOLENT_RATIO_THRESHOLD = config.VIOLENT_RATIO_THRESHOLD
RISKY_RATIO_THRESHOLD = config.RISKY_RATIO_THRESHOLD

# ===============================
# LOAD WINDOW RESULTS
# ===============================
if not os.path.exists(WINDOW_RESULTS_FILE):
    print(f"❌ Window results file not found: {WINDOW_RESULTS_FILE}")
    print("   Please run step7d_window_multimodal_analysis.py first")
    sys.exit(1)

try:
    with open(WINDOW_RESULTS_FILE, "r") as f:
        windows = json.load(f)
except Exception as e:
    print(f"❌ Error reading window results: {e}")
    sys.exit(1)

if not windows:
    print("❌ No window results found in file")
    sys.exit(1)

labels = [w.get("label", "SAFE") for w in windows]
risk_scores = [w.get("risk_score", 0.0) for w in windows]
motions = [w.get("motion", 0.0) for w in windows]

total = len(windows)

if total == 0:
    print("❌ No windows to analyze")
    sys.exit(1)

# ===============================
# BASIC COUNTS
# ===============================
violent_count = labels.count("VIOLENT")
risky_count = labels.count("RISKY")
safe_count = labels.count("SAFE")

violent_ratio = violent_count / total if total > 0 else 0.0
risky_ratio = risky_count / total if total > 0 else 0.0
safe_ratio = safe_count / total if total > 0 else 0.0

# ===============================
# ESCALATION LOGIC
# ===============================
escalation = False
escalation_windows = []
if len(risk_scores) > 1:
    for i in range(len(risk_scores) - 1):
        risk_change = risk_scores[i+1] - risk_scores[i]
        if risk_change > ESCALATION_THRESHOLD:
            escalation = True
            escalation_windows.append({
                "from_window": i,
                "to_window": i+1,
                "risk_increase": round(risk_change, 2)
            })

# ===============================
# STATISTICS
# ===============================
avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
max_risk = float(np.max(risk_scores)) if risk_scores else 0.0
avg_motion = float(np.mean(motions)) if motions else 0.0

# ===============================
# FINAL DECISION
# ===============================
if violent_ratio >= VIOLENT_RATIO_THRESHOLD or escalation:
    final_label = "VIOLENT"
    confidence = min(round(0.7 + violent_ratio * 0.2, 2), 0.99)
    if escalation:
        reason = f"Temporal escalation detected ({len(escalation_windows)} transitions). "
    else:
        reason = ""
    reason += f"Violent content detected in {violent_count}/{total} windows ({violent_ratio*100:.1f}%)"

elif risky_ratio >= RISKY_RATIO_THRESHOLD:
    final_label = "RISKY"
    confidence = min(round(0.6 + risky_ratio * 0.3, 2), 0.99)
    reason = f"Suspicious motion or objects detected in {risky_count}/{total} windows ({risky_ratio*100:.1f}%)"

else:
    final_label = "SAFE"
    confidence = min(round(0.85 + safe_ratio * 0.1, 2), 0.99)
    reason = f"No violent escalation detected. {safe_count}/{total} windows safe ({safe_ratio*100:.1f}%)"

# ===============================
# OUTPUT
# ===============================
final_output = {
    "final_label": final_label,
    "confidence": confidence,
    "reason": reason,
    "statistics": {
        "total_windows": total,
        "violent_windows": violent_count,
        "risky_windows": risky_count,
        "safe_windows": safe_count,
        "violent_ratio": round(violent_ratio, 3),
        "risky_ratio": round(risky_ratio, 3),
        "safe_ratio": round(safe_ratio, 3),
        "avg_risk_score": round(avg_risk, 3),
        "max_risk_score": round(max_risk, 3),
        "avg_motion": round(avg_motion, 3),
        "escalation_detected": escalation,
        "escalation_transitions": escalation_windows
    },
    "timeline": labels
}

print("\n" + "=" * 60)
print("🎬 FINAL VIDEO MODERATION RESULT")
print("=" * 60)
print(json.dumps(final_output, indent=2))
print("=" * 60)

# Save final output - use same directory as window results
output_dir = os.path.dirname(WINDOW_RESULTS_FILE) if os.path.dirname(WINDOW_RESULTS_FILE) else "."
output_file = os.path.join(output_dir, "final_moderation_result.json")
try:
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✅ Final result saved to {output_file}")
except Exception as e:
    print(f"⚠️  Warning: Could not save final result: {e}")
    # Fallback to root directory
    fallback_file = "final_moderation_result.json"
    try:
        with open(fallback_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        print(f"   Saved to fallback location: {fallback_file}")
    except:
        pass