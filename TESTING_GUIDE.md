# Model Testing Guide

Test your video moderation model on YouTube videos and get accuracy reports.

## Quick Start

### 1. Install yt-dlp
```bash
pip install yt-dlp
```

### 2. Configure Test Videos
Edit `test_videos_config.json` and add YouTube URLs with ground truth labels:

```json
{
  "videos": [
    {
      "id": "safe_cooking_1",
      "url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "label": "SAFE",
      "category": "cooking"
    },
    {
      "id": "violent_fight_1",
      "url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "label": "VIOLENT",
      "category": "violence"
    }
  ]
}
```

**Labels:** `SAFE`, `RISKY`, or `VIOLENT`

### 3. Run Tests
```bash
python test_youtube_videos.py
```

## How It Works

1. **Downloads videos temporarily** (first 15 seconds only)
2. **Runs full pipeline** on each video
3. **Compares predictions** with ground truth labels
4. **Generates accuracy report** with:
   - Overall accuracy
   - Per-class metrics (Precision, Recall, F1)
   - Confusion matrix
   - Per-category breakdown

## Output

### Console Output
- Real-time progress for each video
- Overall accuracy percentage
- Per-class metrics
- Confusion matrix
- Per-category accuracy

### JSON Report
Saved to `accuracy_report.json`:
```json
{
  "summary": {
    "total_videos": 10,
    "accuracy": 0.85,
    "correct": 8
  },
  "per_class_metrics": {
    "SAFE": {
      "precision": 0.90,
      "recall": 0.85,
      "f1": 0.87,
      "support": 4
    }
  },
  "confusion_matrix": {...},
  "detailed_results": [...]
}
```

## Finding Test Videos

### Safe Videos (SAFE)
- Cooking tutorials
- Educational content
- Nature documentaries
- Daily life vlogs
- Product reviews

### Risky Videos (RISKY)
- Action movie clips
- Sports with contact
- Stunt videos
- Intense scenes

### Violent Videos (VIOLENT)
- Fight scenes
- Weapon usage
- Violent content
- Aggressive behavior

## Tips

1. **Use diverse categories** - Test different types of content
2. **Balance your dataset** - Equal number of SAFE/RISKY/VIOLENT
3. **Use 15-second clips** - Script automatically uses first 15 seconds
4. **Verify ground truth** - Make sure labels are accurate
5. **Test edge cases** - Kitchen knives (should be SAFE), sports (might be RISKY)

## Example Test Configuration

```json
{
  "videos": [
    {
      "id": "safe_kitchen_knife",
      "url": "https://www.youtube.com/watch?v=COOKING_VIDEO",
      "label": "SAFE",
      "category": "cooking",
      "description": "Cooking with knife - should be SAFE (context matters!)"
    },
    {
      "id": "risky_martial_arts",
      "url": "https://www.youtube.com/watch?v=MARTIAL_ARTS",
      "label": "RISKY",
      "category": "sports",
      "description": "Martial arts - might be RISKY"
    },
    {
      "id": "violent_fight",
      "url": "https://www.youtube.com/watch?v=FIGHT_SCENE",
      "label": "VIOLENT",
      "category": "violence",
      "description": "Fight scene - should be VIOLENT"
    }
  ]
}
```

## Understanding Metrics

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Of videos predicted as X, how many were actually X?
- **Recall**: Of videos that are actually X, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows where model makes mistakes

## Troubleshooting

**yt-dlp not found:**
```bash
pip install yt-dlp
```

**Video download fails:**
- Check if URL is valid
- Video might be private/restricted
- Try different video

**Pipeline fails:**
- Check if RAM checkpoint exists
- Ensure all dependencies installed
- Check video format (MP4 preferred)

**Low accuracy:**
- Review confusion matrix to see where mistakes happen
- Check if ground truth labels are correct
- Test with more diverse videos
- Adjust thresholds in `config.py`
