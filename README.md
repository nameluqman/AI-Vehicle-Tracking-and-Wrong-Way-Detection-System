# 🚀 AI-Driven Intelligent Traffic & Wrong-Way Detection System

> **Enterprise-grade real-time computer vision platform for intelligent traffic monitoring, multi-lane wrong-way violation detection, and cross-stream vehicle re-identification.**

An enterprise-grade, real-time computer vision and deep learning platform engineered for autonomous traffic monitoring, multi-lane wrong-way violation detection, and cross-stream vehicle re-identification.

---

## 📌 System Architecture & Pipeline

```text
       [ Input Video File / Live Video Stream ]
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│               1. ORB Camera Stabilizer                 │
│  (Neutralizes camera bumps, vibrations, and drift)     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│           2. YOLOv11n + ByteTrack Engine               │
│  (Detects target vehicles and tracks persistent IDs)   │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│       3. CLIP Re-ID Extractor & Qdrant Vector DB       │
│  (Computes 512-D embeddings for cross-stream ReID)     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│        4. ROI Polygon & Edge Boundary Filtering        │
│  (Isolates active roadways & filters background noise) │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│       5. Trajectory History & EMA Smoothing             │
│  (Smooths center points to calculate vectors & speed)  │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│   6. K-Means Traffic Calibration & Adaptive Detection  │
│  (Clusters initial flows & detects inverse movement)   │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│   7. Violation Flagging & Evidence Archiving           │
│  (Captures high-res JPEGs & builds structured JSON logs)│
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│          8. OpenCV Rendering & HUD Overlays            │
│  (Draws persistent Red boxes for violators & HUD)      │
└─────────────────────────┬──────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
┌──────────────────┐             ┌─────────────────────┐
│  Processed Video │             │  Live MJPEG Stream  │
│    File (.mp4)   │             │   (Web Dashboard)   │
└──────────────────┘             └─────────────────────┘
```

---

# 🌟 Core Features

### 1. 🎥 ORB Camera Stabilization

Eliminates false motion triggers caused by wind drift, camera shake, or structural vibrations.

### 2. 🚗 YOLOv11n & ByteTrack

Real-time multi-class vehicle detection covering:

* Cars
* Trucks
* Buses
* Motorcycles

ByteTrack provides robust tracking persistence across occlusions.

### 3. 🧠 CLIP Visual Re-Identification

Leverages vision-language embeddings to uniquely identify vehicle appearances across independent streams and camera angles.

### 4. 🗄️ Embedded Qdrant Vector Database

Provides high-performance similarity search for fast, persistent vehicle re-identification payload matching.

### 5. 📐 Dynamic ROI & Boundary Filtering

Masks irrelevant regions such as:

* Sidewalks
* Background clutter
* Non-road areas

using polygon mapping and edge clipping checks.

### 6. 📊 Unsupervised K-Means Traffic Calibration

Automatically learns dominant multi-lane directional flows during startup without requiring manual road orientation configuration.

### 7. ⚡ Adaptive Wrong-Way Verification

Uses:

* Exponential Moving Average (EMA) smoothing
* Directional cosine similarity

to catch inverse movement with zero false alarms.

### 8. 📸 Automated Evidence Archiving

Instantly captures high-resolution snapshots and generates structured JSON audit logs for all confirmed violations.

---

# 🛠️ Technology Stack

## Backend

| Technology                 | Purpose                                 |
| -------------------------- | --------------------------------------- |
| **Python 3.10+**           | Core programming language               |
| **FastAPI**                | Async REST API & MJPEG streaming        |
| **PyTorch**                | Deep learning framework                 |
| **Ultralytics YOLOv11**    | Vehicle detection                       |
| **OpenCV**                 | Computer vision & video processing      |
| **ORB**                    | Camera stabilization                    |
| **Open_CLIP**              | Visual embedding extraction             |
| **Qdrant Vector Database** | Vector storage & similarity retrieval   |
| **NumPy / SciPy**          | Numerical and spatial matrix operations |

## Frontend

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| **Next.js**      | Modern server-side rendered UI   |
| **React**        | Interactive dashboard            |
| **TypeScript**   | Type-safe client architecture    |
| **Tailwind CSS** | Responsive utility-first styling |

---

# 📁 Project Directory Structure

```text
AI-Vehicle-Tracking-and-Wrong-Way-Detection-System/
│
├── backend/
│   ├── pipeline.py
│   │   └── Core vision pipeline
│   │       (Stabilizer, ReID, K-Means, Tracking)
│   │
│   ├── main.py
│   │   └── FastAPI application and endpoint routing
│   │
│   └── requirements.txt
│       └── Python package dependencies
│
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   │   └── Tailored UI styling & design tokens
│   │   │
│   │   ├── layout.tsx
│   │   │   └── Root application layout wrapper
│   │   │
│   │   └── page.tsx
│   │       └── Interactive monitoring dashboard & control center
│   │
│   ├── package.json
│   │   └── Frontend npm dependencies & build scripts
│   │
│   ├── tailwind.config.js
│   │   └── Styling configurations
│   │
│   └── tsconfig.json
│       └── TypeScript compiler settings
│
├── .gitignore
│
└── README.md
    └── Project technical documentation
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/nameluqman/AI-Vehicle-Tracking-and-Wrong-Way-Detection-System.git

cd AI-Vehicle-Tracking-and-Wrong-Way-Detection-System
```

---

## 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

### Create and activate virtual environment

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI development server

```bash
uvicorn main:app --reload
```

Backend runs locally at:

```text
http://127.0.0.1:8000
```

---

## 3. Frontend Setup

Open a separate terminal window and navigate to the frontend directory:

```bash
cd frontend
```

### Install packages

```bash
npm install
```

### Start development server

```bash
npm run dev
```

Frontend runs locally at:

```text
http://localhost:3000
```

---

# 🔌 API Documentation

Once the backend service is running, interactive API documentation is available through FastAPI.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 💡 Best Practices for Testing

For optimal wrong-way detection performance:

### 🎥 Use Elevated or Top-Down Footage

Utilize elevated or top-down traffic camera footage.

### 🛣️ Clear Lane Delineation

Ensure clear lane delineation and stable vehicle trajectories.

### 📺 Recommended Resolution

Verify that the camera stream resolution is at least **720p** for high-accuracy CLIP feature extractions.

---

# 🔄 End-to-End Processing Flow

```text
Input Video / Live Stream
          │
          ▼
   ORB Stabilization
          │
          ▼
 YOLOv11n Detection
          │
          ▼
   ByteTrack IDs
          │
          ▼
 CLIP Re-ID Extraction
          │
          ▼
 Qdrant Vector Database
          │
          ▼
 ROI & Boundary Filtering
          │
          ▼
Trajectory History
          │
          ▼
   EMA Smoothing
          │
          ▼
K-Means Traffic Calibration
          │
          ▼
Wrong-Way Verification
          │
          ▼
Violation Detection
          │
          ├───────────────┐
          ▼               ▼
    Evidence           JSON Logs
    Snapshots
          │
          ▼
 OpenCV HUD Rendering
          │
          ├───────────────┐
          ▼               ▼
 Processed Video     Live MJPEG Stream
                          │
                          ▼
                  Web Dashboard
```

---

# 🚦 Intelligent Traffic Monitoring

The system combines multiple computer vision and deep learning components into a unified traffic intelligence pipeline capable of:

* Vehicle detection
* Persistent vehicle tracking
* Cross-stream vehicle re-identification
* Traffic-flow calibration
* Wrong-way movement detection
* ROI-based filtering
* Camera stabilization
* Trajectory analysis
* Evidence capture
* Structured violation logging
* Live dashboard visualization

---

# 📸 Evidence & Violation Archiving

For confirmed violations, the system automatically provides:

```text
┌───────────────────────────────┐
│      Wrong-Way Violation      │
├───────────────────────────────┤
│                               │
│  High-Resolution JPEG         │
│  Vehicle Identification       │
│  Direction Analysis            │
│  Structured JSON Log           │
│                               │
└───────────────────────────────┘
```

This creates a structured record of detected traffic violations for further inspection and analysis.

---

# 🌐 Web Dashboard

The frontend provides a modern monitoring interface built with:

* Next.js
* React
* TypeScript
* Tailwind CSS

The backend exposes the processed traffic stream through an MJPEG stream while the system performs real-time computer vision processing.

---

# 🧩 Core Computer Vision Components

```text
┌──────────────────────────────────────┐
│          Computer Vision Layer       │
├──────────────────────────────────────┤
│                                      │
│  ORB          → Camera Stabilization │
│  YOLOv11n     → Vehicle Detection    │
│  ByteTrack    → Object Tracking      │
│  CLIP         → Visual Re-ID         │
│  Qdrant       → Vector Search        │
│  ROI          → Road Filtering       │
│  EMA          → Trajectory Smoothing │
│  K-Means      → Flow Calibration     │
│                                      │
└──────────────────────────────────────┘
```

---

# 🚀 Quick Start

### Backend

```powershell
cd backend

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn main:app --reload
```

### Frontend

```bash
cd frontend

npm install

npm run dev
```

### Open the application

```text
Frontend:
http://localhost:3000

Backend:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

ReDoc:
http://127.0.0.1:8000/redoc
```

---

# 📄 License

This project is open-source and available under the **MIT License**.

---
