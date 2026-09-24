# AI Vehicle Tracking & Wrong-Way Detection System

An end-to-end **AI-powered computer vision system** for real-time vehicle detection, multi-object tracking, vehicle re-identification (ReID), trajectory analysis, and wrong-way driving detection.

The system combines **YOLO**, **ByteTrack**, **ResNet-18**, **Qdrant Vector Database**, **FastAPI**, and **Next.js / TypeScript** to create a complete intelligent traffic-monitoring solution.

---

## 📌 Project Overview

The **AI Vehicle Tracking & Wrong-Way Detection System** processes traffic video and automatically identifies and tracks vehicles while analyzing their movement direction.

The system is designed to answer:

- Where are the vehicles?
- What type of vehicle is detected?
- Which vehicle is the same vehicle across consecutive frames?
- What is the vehicle's movement direction?
- Does the vehicle match a previously observed vehicle?
- Is the vehicle moving against the expected traffic direction?
- When was the wrong-way event detected?
- Which vehicle caused the violation?

### High-Level Pipeline

```text
                    VIDEO INPUT
                         │
                         ▼
                ┌─────────────────┐
                │  YOLO Detector  │
                │ Vehicle Detection│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ByteTrack    │
                │ Vehicle Tracking│
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Vehicle Bounding Box    Vehicle Trajectory
              │                     │
              ▼                     ▼
        ┌──────────────┐     ┌─────────────────┐
        │  ResNet-18   │     │ Direction       │
        │ Feature      │     │ Analysis        │
        │ Extraction   │     └────────┬────────┘
        └──────┬───────┘              │
               │                      ▼
               ▼              ┌─────────────────┐
        ┌──────────────┐      │ Wrong-Way       │
        │ Qdrant       │      │ Detection       │
        │ Vector DB    │      └────────┬────────┘
        └──────┬───────┘               │
               │                       │
               └───────────┬───────────┘
                           ▼
                  ┌─────────────────┐
                  │     FastAPI      │
                  │     Backend      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Next.js Dashboard│
                  └─────────────────┘
```

---

# 🚀 Main Features

## 1. Vehicle Detection

The system uses **YOLO** for real-time object detection.

It can detect supported road vehicles such as:

- Cars
- Trucks
- Buses
- Motorcycles
- Vans
- Other supported vehicle classes

For every detection, YOLO provides:

```text
Bounding Box
Confidence Score
Class ID
Vehicle Class
```

---

## 2. Multi-Object Tracking

After detecting vehicles, the system uses **ByteTrack** to maintain vehicle identities across frames.

Without tracking:

```text
Frame 1 → Car
Frame 2 → Car
Frame 3 → Car
```

The system cannot reliably determine whether these are the same vehicle.

With ByteTrack:

```text
Frame 1 → Car → ID 1
Frame 2 → Car → ID 1
Frame 3 → Car → ID 1
Frame 4 → Car → ID 1
```

Example:

```text
ID 1 → Car
ID 2 → Truck
ID 3 → Bus
ID 4 → Motorcycle
```

---

## 3. Vehicle Trajectory Tracking

For every tracked vehicle, the system can maintain its position over time.

Example:

```text
Vehicle ID: 12

Frame 100 → (450, 250)
Frame 110 → (452, 270)
Frame 120 → (455, 292)
Frame 130 → (458, 315)
```

These points form a trajectory that can be used to estimate movement direction.

---

## 4. Wrong-Way Detection

Wrong-way detection analyzes the direction in which a tracked vehicle is moving.

For example, if the expected traffic flow is:

```text
       ↓
       ↓
       ↓
       ↓
```

and a vehicle moves:

```text
       ↑
       ↑
       ↑
       ↑
```

the vehicle is moving against the expected direction.

The general pipeline is:

```text
Vehicle Detection
        ↓
Vehicle Tracking
        ↓
Position History
        ↓
Trajectory Calculation
        ↓
Movement Direction
        ↓
Compare With Road Direction
        ↓
Wrong-Way Decision
        ↓
Alert
```

To reduce false positives, direction should be calculated from multiple frames rather than from a single frame.

---

## 5. ResNet-18 Vehicle Re-Identification

The system uses **ResNet-18** as a visual feature extractor for vehicle crops.

Instead of using ResNet-18 only for classification, it can generate feature embeddings representing visual characteristics of a vehicle.

Pipeline:

```text
Vehicle Bounding Box
        ↓
Crop Vehicle
        ↓
Resize / Preprocess
        ↓
ResNet-18
        ↓
Feature Vector
        ↓
L2 Normalization
        ↓
Qdrant
```

A feature vector can conceptually look like:

```text
[
    0.123,
    0.452,
    0.821,
    0.091,
    ...
]
```

---

## 6. Qdrant Vector Database

The system uses **Qdrant** to store and search vehicle embeddings.

Example:

```text
Current Vehicle
      │
      ▼
ResNet-18 Embedding
      │
      ▼
Qdrant Similarity Search
      │
      ├── Vehicle A → 0.93
      ├── Vehicle B → 0.81
      └── Vehicle C → 0.52
```

Vector similarity can be used to support vehicle re-identification across frames or cameras.

---

## 7. Vehicle Re-Identification Workflow

```text
Detected Vehicle
       ↓
Vehicle Crop
       ↓
Feature Extraction
       ↓
ResNet-18 Embedding
       ↓
Normalize Vector
       ↓
Store in Qdrant
       ↓
Future Vehicle
       ↓
Extract New Embedding
       ↓
Search Qdrant
       ↓
Find Potential Match
```

ReID can help when a vehicle temporarily disappears from the camera view and later appears again.

---

## 8. Wrong-Way Alerts

When the system confirms that a tracked vehicle is moving in the opposite direction, it can generate an alert.

Example:

```text
================================
        WRONG-WAY ALERT
================================

Vehicle ID     : 17
Vehicle Type   : Car
Confidence     : 94%
Direction      : Opposite
Status         : WRONG WAY

================================
```

The alert can then be displayed on the frontend dashboard.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────────────────────┐
│                 VIDEO INPUT                 │
│                                             │
│       MP4 / Camera / RTSP Stream            │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                  YOLO                       │
│                                             │
│ Vehicle Detection                           │
│ Bounding Boxes                              │
│ Confidence Scores                           │
│ Vehicle Classes                             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│               ByteTrack                    │
│                                             │
│ Multi-Object Tracking                       │
│ Persistent Vehicle IDs                      │
│ Vehicle Trajectories                        │
└──────────────────────┬──────────────────────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
┌──────────────────────┐ ┌──────────────────────┐
│    ResNet-18         │ │ Direction Analysis   │
│                      │ │                      │
│ Vehicle Embeddings   │ │ Trajectory Analysis │
└──────────┬───────────┘ └──────────┬───────────┘
           │                        │
           ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐
│      Qdrant          │ │  Wrong-Way Detector  │
│                      │ │                      │
│ Vector Storage       │ │ Direction Comparison │
│ Similarity Search    │ │ Alert Generation     │
└──────────┬───────────┘ └──────────┬───────────┘
           │                        │
           └───────────┬────────────┘
                       ▼
              ┌──────────────────┐
              │     FastAPI      │
              │     Backend      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │    Next.js       │
              │    Dashboard     │
              └──────────────────┘
```

---

# 🛠️ Technology Stack

## Backend

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FastAPI | Backend API |
| PyTorch | Deep learning |
| OpenCV | Video processing |
| Ultralytics YOLO | Vehicle detection |
| ByteTrack | Multi-object tracking |
| Torchvision | ResNet-18 |
| Qdrant | Vector database |
| NumPy | Numerical processing |

## Frontend

| Technology | Purpose |
|---|---|
| Next.js | Frontend framework |
| React | UI |
| TypeScript | Type-safe JavaScript |
| Tailwind CSS | Styling |
| Node.js | JavaScript runtime |
| npm | Package manager |

---

# 📁 Project Structure

```text
AI-Vehicle-Tracking-and-Wrong-Way-Detection-System/
├── backend/
│   ├── main.py          # API entry point & real-time server logic
│   ├── pipeline.py      # Core vision, tracking, and Qdrant pipeline
│   └── requirements.txt # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── globals.css  # Global styles
│   │   ├── layout.tsx   # Root layout wrapper
│   │   └── page.tsx     # Live dashboard & controls
│   ├── package.json     # Frontend dependencies & scripts
│   ├── tailwind.config.js
│   └── tsconfig.json
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation

---

# 📋 Requirements

Before running the project, install:

- Python 3.10 or higher
- Node.js 18 or higher
- npm
- Git
- Qdrant
- OpenCV-compatible video support

A GPU is recommended for faster deep-learning inference, but CPU execution is also possible.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/nameluqman/AI-Vehicle-Tracking-and-Wrong-Way-Detection-System.git
cd AI-Vehicle-Tracking-and-Wrong-Way-Detection-System
```

---

# 🐍 Backend Installation

Enter the backend directory:

```bash
cd backend
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 3. Install Python Dependencies

Make sure the virtual environment is activated:

```bash
pip install -r requirements.txt
```

Check installed packages:

```bash
pip list
```

---

# ▶️ 4. Start Backend

If you are inside the `backend` directory:

```bash
uvicorn main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

If you start the server from the project root instead:

```bash
uvicorn backend.main:app --reload
```

---

# 🌐 Frontend Installation

Open a second terminal.

From the project root:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

---

# ▶️ Start Frontend

Run:

```bash
npm run dev
```

The frontend will normally run at:

```text
http://localhost:3000
```

Open the address in your browser.

---

# 🔄 Running the Full Application

You need two terminals.

## Terminal 1 — Backend

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

## Terminal 2 — Frontend

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:3000
```

---


For wrong-way detection testing, traffic videos should ideally have:

- A stable camera
- Clear road lanes
- Multiple vehicles
- Visible vehicle movement
- Minimal camera movement
- Clear separation between traffic directions

An elevated or top-down traffic-camera view is particularly useful because the direction of vehicle movement is easier to determine.

---

# 🚦 Wrong-Way Detection

The wrong-way detection system uses vehicle trajectories.

Suppose normal traffic direction is:

```text
              ↓
              ↓
              ↓
              ↓
        NORMAL TRAFFIC
```

If a tracked vehicle moves:

```text
              ↑
              ↑
              ↑
              ↑
        WRONG-WAY VEHICLE
```

the system can flag that vehicle.

## Direction Calculation

For a tracked vehicle, multiple center points can be collected:

```text
P1 = (400, 200)
P2 = (405, 220)
P3 = (410, 240)
P4 = (415, 260)
```

The movement vector can be estimated as:

```text
Movement Vector = Last Position - First Position
```

Example:

```text
Start = (400, 200)
End   = (415, 260)

Vector = (15, 60)
```

The resulting movement vector is compared with the expected road direction.

---

# 🧮 Cosine Similarity for Direction

The relationship between two vectors can be calculated using cosine similarity:

```text
cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:

```text
A = Expected Road Direction
B = Vehicle Movement Direction
```

Conceptually:

```text
Same Direction

A ↓
B ↓

Similarity → Positive
```

```text
Opposite Direction

A ↓
B ↑

Similarity → Negative
```

A threshold can then be used to classify movement. The exact threshold should be tuned using representative traffic footage.

---

# 🧭 Direction Analysis Over Multiple Frames

A robust system should not classify a vehicle as wrong-way based on one frame.

Instead:

```text
Frame 1
   ↓
Frame 2
   ↓
Frame 3
   ↓
Frame 4
   ↓
Frame 5
   ↓
Calculate Trajectory
   ↓
Calculate Direction
   ↓
Compare With Expected Flow
   ↓
Confirm Wrong-Way
   ↓
Generate Alert
```

This helps reduce false positives.

---

# 🚗 Vehicle Tracking Example

```text
Frame 100

ID 1 → Car
ID 2 → Truck
ID 3 → Bus
```

Next frame:

```text
Frame 101

ID 1 → Car
ID 2 → Truck
ID 3 → Bus
```

Next frame:

```text
Frame 102

ID 1 → Car
ID 2 → Truck
ID 3 → Bus
```

The same IDs allow the system to maintain vehicle trajectories.

---

# 🧠 ResNet-18 Feature Extraction

For every selected vehicle crop:

```text
Vehicle
   ↓
Bounding Box
   ↓
Crop
   ↓
Resize
   ↓
Normalize
   ↓
ResNet-18
   ↓
Feature Vector
   ↓
L2 Normalization
```

The resulting embedding represents visual characteristics of the vehicle.

---

# 🗄️ Qdrant Vector Search

The vector database stores vehicle embeddings.

Example:

```text
Vehicle A
Embedding A
     │
     ▼
   Qdrant
     │
     ├── Vehicle B → Similarity 0.92
     ├── Vehicle C → Similarity 0.78
     └── Vehicle D → Similarity 0.43
```

This allows the system to retrieve visually similar vehicles.

---

# 🔁 Re-Identification Workflow

```text
New Vehicle
     │
     ▼
Extract Crop
     │
     ▼
ResNet-18
     │
     ▼
Embedding
     │
     ▼
Qdrant Search
     │
     ▼
Find Similar Vehicle
     │
     ▼
Compare Similarity
```

---

# 📸 Vehicle Screenshots

The system can capture vehicle screenshots when specific events occur.

Example:

```text
Vehicle Detected
       ↓
Track Vehicle
       ↓
Wrong-Way Confirmed
       ↓
Capture Screenshot
       ↓
Save Screenshot
       ↓
Display In Dashboard
```

Example directory:

```text
backend/
└── screenshots/
    ├── vehicle_1.jpg
    ├── vehicle_7.jpg
    └── vehicle_17.jpg
```

---

# 🌐 Frontend Dashboard

The Next.js dashboard can provide a visual interface for the AI system.

Typical information includes:

### Live Video

```text
┌──────────────────────────────────┐
│                                  │
│          TRAFFIC VIDEO           │
│                                  │
│     Vehicle 1   Vehicle 2        │
│                                  │
│            Vehicle 3             │
│                                  │
└──────────────────────────────────┘
```

### Vehicle Information

```text
Vehicle ID: 12
Class: Car
Confidence: 0.94
Status: Tracking
```

### Wrong-Way Alert

```text
┌─────────────────────────────┐
│      WRONG-WAY ALERT        │
├─────────────────────────────┤
│ Vehicle ID: 17             │
│ Type: Car                  │
│ Direction: Opposite        │
│ Confidence: 94%            │
└─────────────────────────────┘
```

---

# 🔌 FastAPI Backend

FastAPI acts as the communication layer between the computer-vision pipeline and frontend.

Architecture:

```text
YOLO
 ↓
ByteTrack
 ↓
ReID
 ↓
Wrong-Way Detection
 ↓
FastAPI
 ↓
Next.js
```

FastAPI can provide endpoints for:

- Video status
- Detection information
- Tracking information
- Wrong-way alerts
- Screenshots
- Vehicle data

---

# 📡 API Documentation

When the backend is running, open:

```text
http://127.0.0.1:8000/docs
```

FastAPI provides an interactive Swagger interface where available endpoints can be tested.

---

# ⚡ Performance

System performance depends on:

- YOLO model size
- Input resolution
- Video FPS
- Number of vehicles
- CPU/GPU
- ResNet-18 inference frequency
- Qdrant search frequency
- Tracking configuration

For better performance, ReID does not necessarily need to run on every frame.

Example:

```text
Frame 1 → Detection + Tracking
Frame 2 → Detection + Tracking
Frame 3 → Detection + Tracking
Frame 4 → Detection + Tracking + ReID
Frame 5 → Detection + Tracking
Frame 6 → Detection + Tracking
Frame 7 → Detection + Tracking + ReID
```

---

# 🖥️ CPU and GPU

The system can run on CPU, but GPU acceleration is recommended for real-time performance.

Check CUDA availability:

```python
import torch

print(torch.cuda.is_available())
```

If the result is:

```text
True
```

PyTorch can use a compatible CUDA device.

If:

```text
False
```

the system can still run using CPU.

---

# 🧪 Testing

A useful test video should contain:

```text
Normal Vehicles

Car       → → →
Truck     → → →
Bus       → → →

Wrong-Way Vehicle

Car       ← ← ←
```

The system should:

1. Detect vehicles.
2. Assign tracking IDs.
3. Maintain IDs across frames.
4. Record vehicle trajectories.
5. Calculate movement direction.
6. Compare movement with expected traffic direction.
7. Detect the wrong-way vehicle.
8. Generate an alert.
9. Display the event on the dashboard.

---

# 🧪 Example Detection Result

```json
{
  "track_id": 17,
  "vehicle_class": "car",
  "confidence": 0.94,
  "direction": "opposite",
  "wrong_way": true
}
```

Example frontend representation:

```text
================================
        WRONG-WAY ALERT
================================

Vehicle ID    : 17
Vehicle Type  : Car
Confidence    : 94%
Direction     : Opposite
Wrong Way     : YES

================================
```

---

# 🔍 Troubleshooting

## Backend Does Not Start

Check Python:

```bash
python --version
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn main:app --reload
```

---

## `ModuleNotFoundError`

If you receive:

```text
ModuleNotFoundError: No module named 'ultralytics'
```

install:

```bash
pip install ultralytics
```

Or reinstall all dependencies:

```bash
pip install -r requirements.txt
```

---

## Frontend Does Not Start

Check Node:

```bash
node --version
```

Check npm:

```bash
npm --version
```

Install:

```bash
npm install
```

Run:

```bash
npm run dev
```

---

## Video Does Not Open

Check the video:

```python
import cv2

video = cv2.VideoCapture("videos/traffic.mp4")

print("Opened:", video.isOpened())
print("Frames:", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS:", video.get(cv2.CAP_PROP_FPS))
print("Width:", int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height:", int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video.release()
```

If:

```text
Opened: False
```

check:

- File path
- File permissions
- Video format
- File corruption
- OpenCV/FFmpeg installation

---

## Vehicle IDs Change Frequently

Possible causes:

- Vehicle occlusion
- Poor lighting
- Motion blur
- Low FPS
- Small vehicle size
- Low detection confidence
- Vehicles overlapping
- Fast movement
- Camera movement

Possible improvements:

- Tune YOLO confidence.
- Tune ByteTrack parameters.
- Improve video quality.
- Use a better camera angle.
- Use ReID for additional association.
- Increase tracking history.

---

## Wrong-Way False Positives

Wrong-way detection may produce false positives when:

- A vehicle stops.
- A vehicle changes lanes.
- A vehicle turns.
- The camera moves.
- Tracking temporarily jumps.
- The vehicle is heavily occluded.

A better approach is to use a time window:

```text
Vehicle Track
     ↓
Collect Multiple Positions
     ↓
Smooth Trajectory
     ↓
Calculate Direction
     ↓
Compare With Expected Flow
     ↓
Require Multiple Confirmations
     ↓
Generate Alert
```

---

# 🔐 Environment Variables

Sensitive information should be stored in environment variables.

Example:

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

For the frontend:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Do not commit secret keys to GitHub.

---

# 📝 Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
venv/
.venv/
env/

# Environment files

# Node
node_modules/
.next/
out/

# Qdrant
qdrant_db/

# Generated files
screenshots/

# Video files
*.mp4
*.avi
*.mov
*.mkv



---


