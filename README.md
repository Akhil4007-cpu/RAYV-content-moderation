# Vision AI Video Moderation Pipeline

Automated video content moderation system using YOLO, RAM, and BLIP models to detect violent or inappropriate content.

## 🚀 Quick Start

### Step 1: Clone Repository
```bash
git clone https://github.com/Akhil4007-cpu/RAYV-content-moderation.git
cd RAYV-content-moderation
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download RAM Model Checkpoint
1. Download `ram_swin_large_14m.pth` from:
   - [HuggingFace](https://huggingface.co/spaces/xinyu1205/recognize-anything)
2. Create `checkpoints` folder and place file:
   ```bash
   mkdir checkpoints
   # Move ram_swin_large_14m.pth to checkpoints/
   ```

### Step 5: Run Pipeline
```bash
python main_pipeline.py --video your_video.mp4
```

**That's it!** Results will be saved in `outputs/[video_name_timestamp]/`

---

## 📖 Detailed Step-by-Step Guide

### Installation

#### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, CPU works but slower)
- 8GB+ RAM
- 5GB+ free disk space

#### Complete Installation Steps

1. **Navigate to project directory**
   ```bash
   cd vision_ai_clean
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Upgrade pip**
   ```bash
   pip install --upgrade pip
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Download RAM checkpoint**
   - Visit: https://huggingface.co/spaces/xinyu1205/recognize-anything
   - Download `ram_swin_large_14m.pth`
   - Create `checkpoints` folder: `mkdir checkpoints`
   - Place file: `checkpoints/ram_swin_large_14m.pth`

7. **Verify installation**
   - YOLO model auto-downloads on first run
   - BLIP/CLIP models auto-download from HuggingFace

### Usage

#### Basic Usage
```bash
python main_pipeline.py --video input_video.mp4
```

#### What Happens Automatically:
1. ✅ **Outputs folder is cleaned** - Previous results removed
2. ✅ **New session created** - `outputs/video_name_timestamp/`
3. ✅ **Frames extracted** - From video at regular intervals
4. ✅ **Duplicates removed** - Using CLIP embeddings
5. ✅ **Windows created** - Temporal analysis groups
6. ✅ **Multimodal analysis** - YOLO + RAM + BLIP
7. ✅ **Risk assessment** - Advanced scoring system
8. ✅ **Final decision** - SAFE/RISKY/VIOLENT classification

#### Advanced Options
```bash
# Keep previous results (don't clean outputs)
python main_pipeline.py --video input.mp4 --keep-results

# Custom output directory
python main_pipeline.py --video input.mp4 --output-dir outputs/my_custom_folder

# Run specific steps only
python main_pipeline.py --video input.mp4 --step extract --step analyze

# Skip dependency checks
python main_pipeline.py --video input.mp4 --no-checks
```

### Output Structure

After running, check `outputs/[video_name_timestamp]/`:

```
outputs/
└── video_name_20260110_163906/
    ├── frames_raw/              # All extracted frames
    ├── frames_unique/           # Deduplicated frames
    ├── window_results.json      # Per-window analysis
    ├── windows_info.json        # Window structure
    └── final_moderation_result.json  # Final decision
```

### Understanding Results

#### Final Result JSON
```json
{
  "final_label": "SAFE",
  "confidence": 0.95,
  "risk_score": 0.15,
  "reason": "Low risk across all windows",
  "statistics": {
    "total_windows": 10,
    "safe_windows": 10,
    "risky_windows": 0,
    "violent_windows": 0
  }
}
```

**Labels:**
- `SAFE` - No concerning content detected
- `RISKY` - Some concerning elements, review recommended
- `VIOLENT` - Clear violent/inappropriate content detected

### Configuration

Edit `config.py` to customize:
- **Device**: CPU/GPU selection
- **Model paths**: Custom model locations
- **Risk thresholds**: Sensitivity adjustments
- **Window size**: Temporal analysis granularity
- **Batch size**: Processing speed vs memory

### Troubleshooting

**Problem**: Import errors
- **Solution**: Activate virtual environment: `venv\Scripts\activate`

**Problem**: RAM model not found
- **Solution**: Download `ram_swin_large_14m.pth` to `checkpoints/`

**Problem**: Out of memory
- **Solution**: Reduce `BATCH_SIZE` in `config.py` or use CPU

**Problem**: Slow processing
- **Solution**: Use GPU (CUDA) - set `DEVICE = "cuda"` in `config.py`

**Problem**: Previous results not cleaned
- **Solution**: Results auto-clean on each run. Use `--keep-results` to preserve.

---

## 🏗️ Project Structure

### Core Files
- `main_pipeline.py` - **Main entry point** (run this!)
- `config.py` - All configuration settings
- `model_manager.py` - Shared model loading (no duplication)
- `risk_scorer.py` - Risk assessment logic
- `action_recognition.py` - Action pattern detection

### Pipeline Steps
- `step7a_extract_frames.py` - Extract frames from video
- `step7b_remove_duplicates.py` - Remove duplicate frames (CLIP)
- `step7c_build_windows.py` - Create temporal windows
- `step7d_window_multimodal_analysis.py` - Analyze with YOLO/RAM/BLIP
- `step8_video_decision_engine.py` - Final moderation decision

---

## 🧪 Testing & Evaluation

### Automated YouTube Testing
Test your model on real YouTube videos automatically:

```bash
# Install testing dependency
pip install yt-dlp

# Run automated tests (searches YouTube, tests all categories)
python auto_test_all_categories.py
```

This script:
- ✅ Automatically searches YouTube for videos in all categories
- ✅ Tests SAFE, RISKY, and VIOLENT content
- ✅ Downloads 15-second clips temporarily
- ✅ Runs full pipeline on each video
- ✅ Generates comprehensive accuracy report
- ✅ Provides per-category breakdown

**Test Coverage:**
- **SAFE:** Cooking, Education, Nature, Daily Life, Product Reviews (23 videos)
- **RISKY:** Action Movies, Sports, Stunts (13 videos)
- **VIOLENT:** Fight Scenes, Weapon Usage, Violence (15 videos)
- **Total:** ~51 videos across 11 categories

**Output:**
- `accuracy_report.json` - Detailed JSON report
- `ACCURACY_REPORT.md` - Human-readable markdown report

## 🔬 How It Works

1. **Frame Extraction** - Extracts frames at regular intervals (0.5s default)
2. **Deduplication** - Removes similar frames using CLIP embeddings (97% similarity threshold)
3. **Window Creation** - Groups frames into temporal windows (5 frames per window)
4. **Multimodal Analysis** - Each window analyzed with:
   - **YOLO**: Object detection (weapons, persons, vehicles)
   - **RAM**: Scene tagging (context understanding - kitchen, street, etc.)
   - **BLIP**: Image captioning (descriptive text)
5. **Risk Assessment** - Calculates risk score using:
   - Weapon detection
   - Motion analysis (optical flow)
   - Action recognition (fighting, attacking)
   - Context awareness (safe contexts reduce risk)
6. **Decision** - Final SAFE/RISKY/VIOLENT classification

---

## 🎯 Features

- ✅ **Multi-model ensemble** (YOLO + RAM + BLIP)
- ✅ **Temporal analysis** (sliding windows)
- ✅ **Context-aware** risk scoring
- ✅ **Weapon detection** (knife, baseball bat, scissors)
- ✅ **Motion detection** (optical flow)
- ✅ **Action recognition** (fighting, attacking)
- ✅ **Model caching** (fast execution)
- ✅ **Auto-clean outputs** (fresh results every run)
- ✅ **Session management** (organized outputs)

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- CUDA GPU (optional, CPU works but slower)
- 8GB+ RAM
- 5GB+ disk space

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

---

## 📄 License

See individual model licenses:
- YOLO: AGPL-3.0
- RAM: Apache 2.0
- BLIP: BSD-3-Clause

---

## ✅ Repository Already Hosted on GitHub!

**Live Repository:** https://github.com/Akhil4007-cpu/RAYV-content-moderation

### Enhance Your GitHub Repository

1. **Add Repository Description**
   - Go to: https://github.com/Akhil4007-cpu/RAYV-content-moderation
   - Click the ⚙️ (gear icon) next to "About"
   - Add description: "AI-powered video content moderation pipeline using YOLO, RAM, and BLIP models"

2. **Add Repository Topics**
   - Click "Add topics" button
   - Add these topics:
     - `computer-vision`
     - `video-moderation`
     - `yolo`
     - `deep-learning`
     - `ai`
     - `content-moderation`
     - `pytorch`
     - `multimodal-ai`

3. **Add Badges (Optional)**
   Add to top of README.md:
   ```markdown
   ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
   ![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)
   ```

4. **Enable GitHub Pages (Optional)**
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` / `/docs` folder
   - Save

---

## 📱 LinkedIn Post Template

### Option 1: Technical Focus
```
🚀 Excited to share my latest project: Vision AI Video Moderation Pipeline!

A comprehensive AI-powered system that automatically detects violent or inappropriate content in videos using state-of-the-art computer vision models.

🔧 Tech Stack:
• YOLO v8 - Object detection
• RAM (Recognize Anything Model) - Scene understanding
• BLIP - Image captioning
• CLIP - Duplicate frame removal
• PyTorch - Deep learning framework

✨ Key Features:
✅ Multi-model ensemble for robust detection
✅ Temporal analysis with sliding windows
✅ Context-aware risk scoring
✅ Weapon detection (knives, bats, etc.)
✅ Motion analysis using optical flow
✅ Action recognition (fighting, attacking)

The pipeline processes videos through multiple stages:
1. Frame extraction & deduplication
2. Temporal window grouping
3. Multimodal analysis (YOLO + RAM + BLIP)
4. Advanced risk assessment
5. Final moderation decision

Perfect for content platforms, social media moderation, or safety applications.

🔗 Check it out: [GitHub Link]
💻 Built with Python, PyTorch, and modern CV techniques

#AI #ComputerVision #DeepLearning #VideoModeration #YOLO #PyTorch #MachineLearning #OpenSource
```

### Option 2: Problem-Solution Focus
```
🎯 Problem: How do you automatically moderate video content at scale?

💡 Solution: Vision AI Video Moderation Pipeline

I built an AI system that combines multiple state-of-the-art models to automatically detect and classify potentially violent or inappropriate content in videos.

The system uses:
• Object detection to identify weapons and people
• Scene understanding to recognize context
• Image captioning for descriptive analysis
• Temporal analysis to understand actions over time

Result: Automated, accurate, and scalable video moderation.

🔗 GitHub: [Your Link]
📊 Features: Multi-model ensemble, context-aware scoring, real-time processing

#AI #VideoModeration #ComputerVision #DeepLearning #OpenSource
```

### Option 3: Achievement Focus
```
🎉 Just completed my Vision AI Video Moderation Pipeline!

After weeks of development, I've built a comprehensive system that:
✅ Detects weapons and violent actions
✅ Understands scene context (kitchen vs street)
✅ Analyzes temporal patterns
✅ Provides confidence-scored decisions

Built with YOLO, RAM, BLIP, and CLIP models - combining the best of object detection, scene understanding, and natural language processing.

Open source and ready for deployment! 🔗 [GitHub Link]

#AI #ComputerVision #MachineLearning #OpenSource #DeepLearning
```

### Option 4: Short & Impactful
```
🚀 New Project: AI-Powered Video Moderation

Combining YOLO, RAM, and BLIP models to automatically detect violent/inappropriate content in videos.

✅ Multi-model ensemble
✅ Context-aware analysis
✅ Temporal understanding
✅ Open source

Perfect for content platforms and safety applications.

🔗 [GitHub Link]

#AI #ComputerVision #VideoModeration #OpenSource
```

---

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

---

**Made with ❤️ for automated content moderation**
