import cv2
import time
import math
import os
import uuid
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Initialize embedded local Qdrant instance
global_qdrant_client = QdrantClient(path="./qdrant_db")
EMBEDDING_DIM = 512
GLOBAL_COLLECTION_NAME = "global_vehicle_reid"


# -----------------------------------------------------------------------------
# GlobalIdentityManager
# -----------------------------------------------------------------------------
class GlobalIdentityManager:
    """Generates unique, non-repeating global vehicle identifiers across all videos."""
    def __init__(self, starting_id: int = 1000):
        self._current_id = starting_id

    def mint_new_id(self) -> int:
        self._current_id += 1
        return self._current_id


# -----------------------------------------------------------------------------
# SequentialIDMapper
# -----------------------------------------------------------------------------
class SequentialIDMapper:
    """Maps internal tracking IDs to consecutive display numbers across multiple videos."""
    def __init__(self):
        self.raw_to_seq = {}
        self.next_display_id = 1

    def get_seq_id(self, raw_id: int) -> int:
        if raw_id not in self.raw_to_seq:
            self.raw_to_seq[raw_id] = self.next_display_id
            self.next_display_id += 1
        return self.raw_to_seq[raw_id]


# Persistent global singletons across multiple video pipeline runs
persistent_id_generator = GlobalIdentityManager(starting_id=1000)
persistent_id_mapper = SequentialIDMapper()


# -----------------------------------------------------------------------------
# RealVehicleReIDExtractor
# -----------------------------------------------------------------------------
class RealVehicleReIDExtractor:
    """Extracts 512-D normalized visual feature embeddings using ResNet-18."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1]).to(self.device)
        self.feature_extractor.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, crop_image: np.ndarray) -> list:
        if crop_image.size == 0 or crop_image.shape[0] < 40 or crop_image.shape[1] < 40:
            return None

        img_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(tensor_img).squeeze().cpu().numpy()

        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features.tolist()[:EMBEDDING_DIM]


# -----------------------------------------------------------------------------
# ensure_global_qdrant_collection
# -----------------------------------------------------------------------------
def ensure_global_qdrant_collection():
    """Creates single unified Qdrant vector store if it does not exist."""
    collections = global_qdrant_client.get_collections().collections
    if not any(c.name == GLOBAL_COLLECTION_NAME for c in collections):
        global_qdrant_client.create_collection(
            collection_name=GLOBAL_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )


# -----------------------------------------------------------------------------
# resolve_vehicle_identity
# -----------------------------------------------------------------------------
def resolve_vehicle_identity(
    crop: np.ndarray,
    reid_extractor: RealVehicleReIDExtractor,
    frame_count: int,
    similarity_threshold: float = 0.82
) -> int:
    """Extracts embedding, queries unified Qdrant store, and assigns persistent identity."""
    embedding = reid_extractor.extract_embedding(crop)
    if embedding is None:
        return None

    search_response = global_qdrant_client.query_points(
        collection_name=GLOBAL_COLLECTION_NAME,
        query=embedding,
        limit=1,
        score_threshold=similarity_threshold,
        with_payload=True
    )

    if search_response.points and search_response.points[0].payload:
        return search_response.points[0].payload["track_id"]

    assigned_id = persistent_id_generator.mint_new_id()

    global_qdrant_client.upsert(
        collection_name=GLOBAL_COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"track_id": assigned_id, "frame": frame_count}
            )
        ]
    )
    return assigned_id


# -----------------------------------------------------------------------------
# process_frame
# -----------------------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    frame_count: int,
    yolo_model: YOLO,
    reid_extractor: RealVehicleReIDExtractor,
    track_history: dict,
    violation_counter: dict,
    flagged_tracks: set,
    indexed_vehicles: set,
    resolved_id_map: dict,
    calibration_vectors: list,
    dominant_flow_vectors: list,
    calibration_frames: int,
    video_id: str,
    violation_dir: str,
    current_fps: float = 0.0
) -> tuple:
    """Core tracking, Re-ID, trajectory smoothing, and direction violation evaluation."""
    height, width = frame.shape[:2]
    detected_violations = []

    roi_polygon = np.array([
        [int(width * 0.05) + 20, int(height * 0.25) + 20],
        [int(width * 0.95) - 20, int(height * 0.25) + 20],
        [int(width * 0.98) - 20, int(height * 0.95) - 20],
        [int(width * 0.02) + 20, int(height * 0.95) - 20]
    ], np.int32)

    results = yolo_model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            is_clipping_edge = (x1 <= 15 or y1 <= 15 or x2 >= width - 15 or y2 >= height - 15)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            crop = frame[y1_c:y2_c, x1_c:x2_c]

            if track_id not in indexed_vehicles:
                if crop.size > 0 and crop.shape[0] >= 40 and crop.shape[1] >= 40:
                    resolved_id = resolve_vehicle_identity(
                        crop, reid_extractor, frame_count, similarity_threshold=0.82
                    )
                    if resolved_id is not None:
                        resolved_id_map[track_id] = resolved_id
                        indexed_vehicles.add(track_id)
                    else:
                        continue
                else:
                    continue

            if track_id not in resolved_id_map:
                continue

            active_id = resolved_id_map[track_id]
            display_id = persistent_id_mapper.get_seq_id(active_id)
            raw_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if active_id not in track_history:
                track_history[active_id] = []
                violation_counter[active_id] = 0

            if is_clipping_edge:
                violation_counter[active_id] = 0

            if len(track_history[active_id]) > 0:
                prev_center = track_history[active_id][-1]
                smoothed_x = int(0.35 * raw_center[0] + 0.65 * prev_center[0])
                smoothed_y = int(0.35 * raw_center[1] + 0.65 * prev_center[1])
                center = (smoothed_x, smoothed_y)
            else:
                center = raw_center

            track_history[active_id].append(center)
            if len(track_history[active_id]) > 30:
                track_history[active_id].pop(0)

            is_wrong_way = False
            is_inside_roi = cv2.pointPolygonTest(roi_polygon, (float(center[0]), float(center[1])), False) >= 0

            if not is_clipping_edge and is_inside_roi and len(track_history[active_id]) >= 20:
                p_start = track_history[active_id][0]
                p_end = track_history[active_id][-1]

                dx = float(p_end[0] - p_start[0])
                dy = float(p_end[1] - p_start[1])
                dist = math.hypot(dx, dy)
                avg_speed_per_frame = dist / len(track_history[active_id])

                if dist >= 130.0 and avg_speed_per_frame > 2.5:
                    unit_vector = np.array([dx / dist, dy / dist], dtype=np.float64)

                    if frame_count <= calibration_frames:
                        calibration_vectors.append(unit_vector)
                    elif len(dominant_flow_vectors) > 0:
                        similarities = [np.dot(unit_vector, flow) for flow in dominant_flow_vectors]
                        max_similarity = max(similarities)

                        if max_similarity < -0.939:
                            violation_counter[active_id] += 1
                        else:
                            violation_counter[active_id] = 0

                        if violation_counter[active_id] >= 15:
                            is_wrong_way = True
                            if active_id not in flagged_tracks:
                                flagged_tracks.add(active_id)

                                if crop.size > 0:
                                    filename = f"wrongway_track_{display_id}_frame_{frame_count}.jpg"
                                    crop_filepath = os.path.join(violation_dir, filename)
                                    cv2.imwrite(crop_filepath, np.ascontiguousarray(crop))

                                    detected_violations.append({
                                        "id": f"{video_id}_{display_id}_{frame_count}",
                                        "video_id": video_id,
                                        "track_id": display_id,
                                        "filename": filename,
                                        "image_url": f"/media/violations_{video_id}/{filename}",
                                        "timestamp_frame": frame_count,
                                        "violation_type": "Wrong Way Driving"
                                    })
                else:
                    violation_counter[active_id] = 0
            else:
                violation_counter[active_id] = 0

            if is_wrong_way:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame, f"WRONG WAY #{display_id}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID: {display_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    if frame_count == calibration_frames and len(calibration_vectors) > 0:
        valid_vecs = np.array([v for v in calibration_vectors if isinstance(v, np.ndarray) and v.shape == (2,)])
        if len(valid_vecs) >= 15:
            mean_vec = np.mean(valid_vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0.5:
                dominant_flow_vectors.append(mean_vec / norm)

            if len(dominant_flow_vectors) > 0:
                opp_vecs = [v for v in valid_vecs if np.dot(v, dominant_flow_vectors[0]) < -0.3]
                if len(opp_vecs) > len(valid_vecs) * 0.15:
                    opp_mean = np.mean(opp_vecs, axis=0)
                    opp_norm = np.linalg.norm(opp_mean)
                    if opp_norm > 0:
                        dominant_flow_vectors.append(opp_mean / opp_norm)

    status_str = "CALIBRATING" if frame_count <= calibration_frames else "ACTIVE"
    height, width = frame.shape[:2]

    # Status Text (Top Left)
    cv2.putText(
        frame,
        f"Frame: {frame_count} | Status: {status_str}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # FPS Overlay (Top Right)
    cv2.putText(
        frame,
        f"FPS: {current_fps:.1f}",
        (width - 160, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return frame, detected_violations


# -----------------------------------------------------------------------------
# process_video_pipeline
# -----------------------------------------------------------------------------
def process_video_pipeline(video_id: str, input_path: str, output_path: str, violation_dir: str):
    os.makedirs(violation_dir, exist_ok=True)
    ensure_global_qdrant_collection()

    yolo_model = YOLO("yolo11n.pt")
    reid_extractor = RealVehicleReIDExtractor()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_history = {}
    violation_counter = {}
    flagged_tracks = set()
    indexed_vehicles = set()
    resolved_id_map = {}
    detected_violations = []

    calibration_vectors = []
    dominant_flow_vectors = []
    calibration_frames = 90
    frame_count = 0

    # FPS Timing tracking variables
    prev_time = time.time()
    current_fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        now = time.time()
        time_diff = now - prev_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
        prev_time = now

        frame_count += 1

        frame, violations = process_frame(
            frame, frame_count, yolo_model, reid_extractor, track_history,
            violation_counter, flagged_tracks, indexed_vehicles, resolved_id_map,
            calibration_vectors, dominant_flow_vectors, calibration_frames, video_id, violation_dir,
            current_fps=current_fps
        )
        detected_violations.extend(violations)
        out.write(frame)

    cap.release()
    out.release()
    return detected_violations


# -----------------------------------------------------------------------------
# stream_video_pipeline
# -----------------------------------------------------------------------------
def stream_video_pipeline(video_id: str, input_path: str, violation_dir: str):
    """Generator function to stream live MJPEG encoded frames to HTTP clients."""
    os.makedirs(violation_dir, exist_ok=True)
    ensure_global_qdrant_collection()

    yolo_model = YOLO("yolo11n.pt")
    reid_extractor = RealVehicleReIDExtractor()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {input_path}")

    track_history = {}
    violation_counter = {}
    flagged_tracks = set()
    indexed_vehicles = set()
    resolved_id_map = {}

    calibration_vectors = []
    dominant_flow_vectors = []
    calibration_frames = 90
    frame_count = 0

    # FPS Timing tracking variables
    prev_time = time.time()
    current_fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            now = time.time()
            time_diff = now - prev_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
            prev_time = now

            frame_count += 1
            frame, _ = process_frame(
                frame, frame_count, yolo_model, reid_extractor, track_history,
                violation_counter, flagged_tracks, indexed_vehicles, resolved_id_map,
                calibration_vectors, dominant_flow_vectors, calibration_frames, video_id, violation_dir,
                current_fps=current_fps
            )

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()