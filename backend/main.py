import os
import glob
import shutil
import uuid
import math
import asyncio
import cv2
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchreid")
from typing import List, Optional, Tuple, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from pipeline import process_video_pipeline, stream_video_pipeline

app = FastAPI(title="AI Vehicle Tracking & ReID System", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(STORAGE_DIR, exist_ok=True)

app.mount("/media", StaticFiles(directory=STORAGE_DIR), name="media")

# In-Memory Registries
videos_db = {}
violations_db = []
calibration_db = {}  # Stores custom UI lane calibration points per video_id

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Pending violations queue for each video
        self.pending_violations: Dict[str, list] = {}

    async def connect(self, websocket: WebSocket, video_id: str):
        await websocket.accept()
        if video_id not in self.active_connections:
            self.active_connections[video_id] = []
            self.pending_violations[video_id] = []
        self.active_connections[video_id].append(websocket)
        
        # Send any pending violations
        if self.pending_violations[video_id]:
            for violation in self.pending_violations[video_id]:
                try:
                    await websocket.send_json(violation)
                except:
                    pass
            self.pending_violations[video_id] = []

    def disconnect(self, websocket: WebSocket, video_id: str):
        if video_id in self.active_connections:
            self.active_connections[video_id].remove(websocket)
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]

    async def broadcast_violation(self, video_id: str, violation: dict):
        if video_id in self.active_connections:
            for connection in self.active_connections[video_id]:
                try:
                    await connection.send_json(violation)
                except:
                    pass
        else:
            # Store pending violations if no connections
            if video_id not in self.pending_violations:
                self.pending_violations[video_id] = []
            self.pending_violations[video_id].append(violation)

manager = ConnectionManager()


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class ViolationSchema(BaseModel):
    id: str
    video_id: str
    track_id: int
    filename: str
    image_url: str
    timestamp_frame: int
    violation_type: str


class VideoResponseSchema(BaseModel):
    video_id: str
    original_filename: str
    status: str
    processed_video_url: Optional[str] = None
    violations_count: int = 0


class CalibrationRequestSchema(BaseModel):
    video_id: str
    upward_points: List[Tuple[int, int]]    # [(x1, y1), (x2, y2)]
    downward_points: List[Tuple[int, int]]  # [(x1, y1), (x2, y2)]


class CalibrationResponseSchema(BaseModel):
    status: str
    video_id: str
    calib_info: Dict[str, Any]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_unit_vector(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> np.ndarray:
    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])
    length = math.hypot(dx, dy)
    if length == 0:
        return np.array([0.0, -1.0], dtype=np.float64)
    return np.array([dx / length, dy / length], dtype=np.float64)


def get_default_calibration(input_path: str) -> dict:
    """Fallback calibration computed headlessly from frame dimensions."""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    up_pt1, up_pt2 = (int(width * 0.3), int(height * 0.8)), (int(width * 0.3), int(height * 0.2))
    down_pt1, down_pt2 = (int(width * 0.7), int(height * 0.2)), (int(width * 0.7), int(height * 0.8))

    return {
        "expected_flows": {
            "upward_vec": compute_unit_vector(up_pt1, up_pt2),
            "downward_vec": compute_unit_vector(down_pt1, down_pt2),
            "upward_x": float(up_pt1[0]),
            "downward_x": float(down_pt1[0])
        },
        "lines": {
            "upward": [up_pt1, up_pt2],
            "downward": [down_pt1, down_pt2]
        }
    }


def run_cv_pipeline(video_id: str, input_path: str):
    videos_db[video_id]["status"] = "processing"
    output_video_name = f"processed_{video_id}.mp4"
    output_video_path = os.path.join(STORAGE_DIR, output_video_name)
    violation_folder = os.path.join(STORAGE_DIR, f"violations_{video_id}")
    os.makedirs(violation_folder, exist_ok=True)

    calib_info = calibration_db.get(video_id) or get_default_calibration(input_path)

    try:
        detected_violations = process_video_pipeline(
            video_id=video_id,
            input_path=input_path,
            output_path=output_video_path,
            violation_dir=violation_folder,
            calib_info=calib_info
        )
        
        violations_db.extend(detected_violations)
        
        videos_db[video_id]["status"] = "completed"
        videos_db[video_id]["processed_video_url"] = f"/media/{output_video_name}"
        videos_db[video_id]["violations_count"] = len(detected_violations)
        print(f"[SUCCESS] Processing finished for {video_id}. Total violations: {len(detected_violations)}")
    except Exception as e:
        videos_db[video_id]["status"] = f"failed: {str(e)}"
        print(f"[ERROR] Pipeline failed for {video_id}: {str(e)}")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/v1/videos/upload", response_model=VideoResponseSchema)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported video format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    video_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(file.filename)[1]
    input_file_path = os.path.join(STORAGE_DIR, f"raw_{video_id}{file_extension}")

    try:
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    video_record = {
        "video_id": video_id,
        "original_filename": file.filename,
        "status": "queued",
        "processed_video_url": None,
        "violations_count": 0
    }
    
    videos_db[video_id] = video_record
    background_tasks.add_task(run_cv_pipeline, video_id, input_file_path)
    
    return video_record


@app.get("/api/v1/videos/first-frame/{video_id}")
async def get_first_frame(video_id: str):
    """Extracts and serves frame 1 for canvas calibration on the web frontend."""
    matching_files = glob.glob(os.path.join(STORAGE_DIR, f"raw_{video_id}.*"))
    if not matching_files:
        raise HTTPException(status_code=404, detail="Raw video file not found.")

    frame_path = os.path.join(STORAGE_DIR, f"frame_{video_id}.jpg")
    
    if not os.path.exists(frame_path):
        cap = cv2.VideoCapture(matching_files[0])
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=500, detail="Could not extract initial video frame.")

        cv2.imwrite(frame_path, frame)

    return FileResponse(frame_path, media_type="image/jpeg")


@app.post("/api/v1/videos/calibrate", response_model=CalibrationResponseSchema)
async def save_calibration(config: CalibrationRequestSchema):
    """Receives clicked lane calibration vectors directly from the Web UI."""
    if len(config.upward_points) < 2 or len(config.downward_points) < 2:
        raise HTTPException(status_code=400, detail="Must provide 2 points for both upward and downward lanes.")

    up_pt1, up_pt2 = config.upward_points[0], config.upward_points[1]
    down_pt1, down_pt2 = config.downward_points[0], config.downward_points[1]

    upward_vec = compute_unit_vector(up_pt1, up_pt2)
    downward_vec = compute_unit_vector(down_pt1, down_pt2)

    # Internal pipeline state uses raw NumPy arrays
    calib_info_internal = {
        "expected_flows": {
            "upward_vec": upward_vec,
            "downward_vec": downward_vec,
            "upward_x": (up_pt1[0] + up_pt2[0]) / 2.0,
            "downward_x": (down_pt1[0] + down_pt2[0]) / 2.0
        },
        "lines": {
            "upward": [up_pt1, up_pt2],
            "downward": [down_pt1, down_pt2]
        }
    }

    # Store for process_video_pipeline and stream_video_pipeline
    calibration_db[config.video_id] = calib_info_internal

    # JSON-serializable version for API response (convert ndarray -> list)
    calib_info_response = {
        "expected_flows": {
            "upward_vec": upward_vec.tolist(),
            "downward_vec": downward_vec.tolist(),
            "upward_x": (up_pt1[0] + up_pt2[0]) / 2.0,
            "downward_x": (down_pt1[0] + down_pt2[0]) / 2.0
        },
        "lines": {
            "upward": [up_pt1, up_pt2],
            "downward": [down_pt1, down_pt2]
        }
    }

    return {
        "status": "success",
        "video_id": config.video_id,
        "calib_info": calib_info_response
    }


@app.get("/api/v1/videos/stream/{video_id}")
async def stream_video(video_id: str):
    """Streams the real-time annotated tracking feed via MJPEG."""
    matching_files = glob.glob(os.path.join(STORAGE_DIR, f"raw_{video_id}.*"))
    if not matching_files:
        raise HTTPException(status_code=404, detail="Raw video file not found.")

    input_path = matching_files[0]
    violation_dir = os.path.join(STORAGE_DIR, f"violations_{video_id}")
    os.makedirs(violation_dir, exist_ok=True)

    calib_info = calibration_db.get(video_id) or get_default_calibration(input_path)
    event_loop = asyncio.get_running_loop()

    # Create callback for real-time violation updates via WebSocket
    def violation_callback(violation_record: dict):
        print(f"[Callback] Violation detected: {violation_record}")
        violations_db.append(violation_record)
        video_record = videos_db.get(video_id)
        if video_record:
            video_record["violations_count"] = len({
                violation["id"]
                for violation in violations_db
                if violation.get("video_id") == video_id
            })

        def broadcast():
            asyncio.create_task(manager.broadcast_violation(video_id, violation_record))

        event_loop.call_soon_threadsafe(broadcast)

    return StreamingResponse(
        stream_video_pipeline(
            video_id=video_id,
            input_path=input_path,
            violation_dir=violation_dir,
            calib_info=calib_info,
            enable_reid=True,
            violation_callback=violation_callback
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/v1/videos/{video_id}", response_model=VideoResponseSchema)
async def get_video_status(video_id: str):
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video ID not found.")
    return videos_db[video_id]


@app.websocket("/ws/violations/{video_id}")
async def websocket_violations(websocket: WebSocket, video_id: str):
    """WebSocket endpoint for real-time violation updates."""
    await manager.connect(websocket, video_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, video_id)


@app.get("/api/v1/violations", response_model=List[ViolationSchema])
async def get_violations(video_id: Optional[str] = None):
    if video_id:
        return [v for v in violations_db if v.get("video_id") == video_id]
    return violations_db


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)