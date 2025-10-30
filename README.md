# BasketballAI (Hoopers Den)

AI-assisted basketball training tools for shooting, dribbling, and movement analysis. The project combines pose estimation (YOLOv8-pose) with domain logic for basketball skills, and includes a concept UI (“Hoopers Den”) that outlines app modules, gamification, and trainer workflows.

## ✨ Features

- Pose-based analysis using pretrained pose weights (e.g., `yolov8s-pose.pt`) and a basketball model (`basketballModel.pt`).
- Dribbling & Shooting modules (concept/UI) with metrics like posture, arc, timing, hand switches, rhythm, and more.
- Gamification layer (concept/UI): XP, streaks, badges, coach challenges, leaderboards.

## 🗂️ Repository Layout

BasketballAi/
├─ DribbleAPI/                # (stub/placeholder for dribbling analysis API)
├─ ShotAPI/                   # (stub/placeholder for shooting analysis API)
├─ travel_detection.py        # Pose-based movement logic (e.g., traveling detection)
├─ basketballModel.pt         # Domain model weights
├─ yolov8s-pose.pt            # YOLOv8 pose weights
├─ requirement.txt            # Python dependencies
├─ README.md
└─ *.MOV                      # Example/input video files

> A link to a concept deck/UI (“Hoopers Den”) is in the repo metadata:
> https://basketball-ashen.vercel.app/ppt.html

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A machine with CPU (GPU recommended for real-time)
- (If using YOLOv8 pose) PyTorch + Ultralytics

### Installation

# 1) Clone
git clone https://github.com/gangulya111/BasketballAi.git
cd BasketballAi

# 2) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
# NOTE: the file is named "requirement.txt" in the repo
pip install -r requirement.txt

> If you plan to use YOLOv8-pose directly, ensure Ultralytics is installed:
> pip install ultralytics

## ▶️ Usage

### 1) Run the travel/motion detection script

python travel_detection.py --video path/to/input.mp4

Common flags (if implemented in your script):

--video     path to a video file (or omit to try webcam index 0)
--weights   path to a pose model (defaults to yolov8s-pose.pt)
--save      save annotated video to runs/ (if supported)

> Tip: The repo includes several sample `.MOV` files you can test with.

### 2) Explore the concept UI

Review the product/UI concept for the full app vision, modules, metrics, and gamification:
Hoopers Den – Presentation → https://basketball-ashen.vercel.app/ppt.html

## 📦 Models

- yolov8s-pose.pt: Pose backbone for keypoints/skeletons.
- basketballModel.pt: Custom weights for basketball-specific classification/logic.

> Note: Large model files increase repo size. In production, consider storing them via Git LFS or downloading on first run.

## 🧩 Planned APIs (stubs)

- DribbleAPI/ – HTTP endpoints to run/return dribbling metrics (frequency, hand switches, stance, control).
- ShotAPI/ – HTTP endpoints for shot metrics (elbow angle, arc/entry angle, release timing).

## 📊 Key Metrics (from concept)

- Shooting: posture, elbow angle, arc/entry angle, release speed, timing.
- Dribbling: stance height, dribble height (knee–waist optimal), bounces/sec, transitions, ball proximity control.
- Training: strength/agility drills tracking, completion scores.
- Gamification: XP, streaks, badges, coach challenges, leaderboards.

## 🔧 Development Notes

- Prefer GPU for real-time inference.
- Keep pose and domain logic modular (pose → features → skill scoring).
- For consistent experiments, pin dependency versions in requirement.txt.

## 🗺️ Roadmap

- [ ] Implement REST endpoints under DribbleAPI/ and ShotAPI/.
- [ ] Add end-to-end pipeline: video → pose → metrics → score/feedback.
- [ ] Export JSON summaries for UI consumption.
- [ ] Add tests and CI.
- [ ] Optional: integrate live webcam and file uploads in a simple web UI.

## 🤝 Contributing

PRs and issues are welcome—please open an Issue to discuss significant changes first.
Style: follow PEP8 for Python, include docstrings and small, focused functions.

## 🛡️ License

Add your preferred license (e.g., MIT) to LICENSE and reference it here.

## 🙏 Acknowledgements

- YOLOv8 Pose model for keypoint detection.
- Coaches/players who inspired the metrics referenced in the Hoopers Den concept.

 
 
 https://basketball-ashen.vercel.app/ppt.html
