import os
import glob
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pipeline import process_video_pipeline ,stream_video_pipeline 
# from pipeline import process_video_pipeline, stream_video_pipeline
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
videos_db = {}
violations_db = []
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
def run_cv_pipeline(video_id: str, input_path: str):
    videos_db[video_id]["status"] = "processing"
    output_video_name = f"processed_{video_id}.mp4"
    output_video_path = os.path.join(STORAGE_DIR, output_video_name)
    violation_folder = os.path.join(STORAGE_DIR, f"violations_{video_id}")
    try:
        # Execute active YOLO + ByteTrack + ReID Pipeline
        detected_violations = process_video_pipeline(
            video_id=video_id,
            input_path=input_path,
            output_path=output_video_path,
            violation_dir=violation_folder
        )
        # Append violations to main registry
        violations_db.extend(detected_violations)
        videos_db[video_id]["status"] = "completed"
        videos_db[video_id]["processed_video_url"] = f"/media/{output_video_name}"
        videos_db[video_id]["violations_count"] = len(detected_violations)
        print(f"[SUCCESS] Processing finished for {video_id}. Total violations: {len(detected_violations)}")
    except Exception as e:
        videos_db[video_id]["status"] = f"failed: {str(e)}"
        print(f"[ERROR] Pipeline failed for {video_id}: {str(e)}")
@app.post("/api/v1/videos/upload", response_model=VideoResponseSchema)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    video_id = str(uuid.uuid4())[:8]
    file_extension = os.path.splitext(file.filename)[1]
    input_file_path = os.path.join(STORAGE_DIR, f"raw_{video_id}{file_extension}")
    with open(input_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    videos_db[video_id] = {
        "video_id": video_id,
        "original_filename": file.filename,
        "status": "queued",
        "processed_video_url": None,
        "violations_count": 0
    }
    background_tasks.add_task(run_cv_pipeline, video_id, input_file_path)
    return videos_db[video_id]
@app.get("/api/v1/videos/stream/{video_id}")
async def stream_video(video_id: str):
    """Streams the real-time annotated tracking feed via MJPEG."""
    # Find matching uploaded file format
    matching_files = glob.glob(os.path.join(STORAGE_DIR, f"raw_{video_id}.*"))
    if not matching_files:
        raise HTTPException(status_code=404, detail="Raw video file not found.")
    input_path = matching_files[0]
    violation_dir = os.path.join(STORAGE_DIR, f"violations_{video_id}")
    return StreamingResponse(
        stream_video_pipeline(video_id, input_path, violation_dir),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
@app.get("/api/v1/videos/{video_id}", response_model=VideoResponseSchema)
async def get_video_status(video_id: str):
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video ID not found.")
    return videos_db[video_id]
@app.get("/api/v1/violations", response_model=List[ViolationSchema])
async def get_violations(video_id: Optional[str] = None):
    if video_id:
        return [v for v in violations_db if v["video_id"] == video_id]
    return violations_db
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 