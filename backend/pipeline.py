import os
import time
import math
import uuid
import cv2
import numpy as np
import torch
from PIL import Image
import open_clip
from ultralytics import YOLO
from sklearn.cluster import KMeans
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Initialize embedded Qdrant vector engine
global_qdrant_client = QdrantClient(path="./qdrant_db")
EMBEDDING_DIM = 512
GLOBAL_COLLECTION_NAME = "global_vehicle_reid"


# -----------------------------------------------------------------------------
# Identity Management Classes
# -----------------------------------------------------------------------------
class GlobalIdentityManager:
    """Generates unique, non-repeating global vehicle identifiers across all video streams."""
    def __init__(self, starting_id: int = 1000):
        self._current_id = starting_id

    def mint_new_id(self) -> int:
        self._current_id += 1
        return self._current_id


class SequentialIDMapper:
    """Maps global vehicle identities to clean sequential numbers (1, 2, 3, ...) per video session."""
    def __init__(self):
        self.global_to_seq = {}
        self.next_seq_id = 1

    def get_seq_id(self, global_id: int) -> int:
        if global_id not in self.global_to_seq:
            self.global_to_seq[global_id] = self.next_seq_id
            self.next_seq_id += 1
        return self.global_to_seq[global_id]


persistent_id_generator = GlobalIdentityManager(starting_id=1000)


# -----------------------------------------------------------------------------
# ORB Camera Stabilizer (Mitigates Camera Bumps, Tilts, and Drift)
# -----------------------------------------------------------------------------
class ORBStabilizer:
    """Aligns incoming frames to a stable reference frame using ORB feature matching and affine warping."""
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=750)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_gray = None
        self.ref_kp = None
        self.ref_des = None

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.ref_gray is None:
            self.ref_gray = gray
            self.ref_kp = kp
            self.ref_des = des
            return frame

        if des is None or self.ref_des is None or len(kp) < 15 or len(self.ref_kp) < 15:
            return frame

        matches = self.bf.match(self.ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 15:
            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in matches[:25]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches[:25]]).reshape(-1, 1, 2)

            M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
            if M is not None:
                h, w = frame.shape[:2]
                stabilized = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                return stabilized

        return frame


# -----------------------------------------------------------------------------
# CLIP Re-ID Extractor (Lazy Singleton)
# -----------------------------------------------------------------------------
class CLIPVehicleReIDExtractor:
    """Extracts 512-D normalized visual feature embeddings using CLIP ViT-B/32."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract_embedding(self, crop_image: np.ndarray) -> list:
        if crop_image.size == 0 or crop_image.shape[0] < 50 or crop_image.shape[1] < 50:
            return None

        img_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        image_features = self.model.encode_image(tensor_img).squeeze().cpu().numpy()

        if image_features.ndim > 1:
            image_features = image_features.flatten()

        norm = np.linalg.norm(image_features)
        if norm > 0:
            image_features = image_features / norm

        return image_features.tolist()[:EMBEDDING_DIM]


GLOBAL_CLIP_EXTRACTOR = None

def get_clip_extractor():
    global GLOBAL_CLIP_EXTRACTOR
    if GLOBAL_CLIP_EXTRACTOR is None:
        GLOBAL_CLIP_EXTRACTOR = CLIPVehicleReIDExtractor()
    return GLOBAL_CLIP_EXTRACTOR


# -----------------------------------------------------------------------------
# Qdrant Database Setup
# -----------------------------------------------------------------------------
def ensure_global_qdrant_collection():
    """Ensures vector database collection exists with cosine distance metric."""
    collections = global_qdrant_client.get_collections().collections
    if not any(c.name == GLOBAL_COLLECTION_NAME for c in collections):
        global_qdrant_client.create_collection(
            collection_name=GLOBAL_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )


def resolve_vehicle_identity(
    crop: np.ndarray,
    reid_extractor: CLIPVehicleReIDExtractor,
    frame_count: int,
    currently_active_global_ids: set,
    similarity_threshold: float = 0.86
) -> int:
    """Matches candidate embeddings against Qdrant index or mints a new non-repeating ID."""
    if crop.shape[0] < 50 or crop.shape[1] < 50:
        return None

    embedding = reid_extractor.extract_embedding(crop)
    if embedding is None:
        return None

    search_response = global_qdrant_client.query_points(
        collection_name=GLOBAL_COLLECTION_NAME,
        query=embedding,
        limit=5,
        score_threshold=similarity_threshold,
        with_payload=True
    )

    if search_response.points:
        for point in search_response.points:
            if point.payload:
                candidate_id = point.payload.get("track_id")
                if candidate_id and candidate_id not in currently_active_global_ids:
                    return candidate_id

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
# K-Means Multi-Directional Traffic Flow Calibration
# -----------------------------------------------------------------------------
def compute_dominant_traffic_flows(calibration_vectors: list, max_flows: int = 4) -> list:
    """
    Uses K-Means clustering to extract multiple distinct traffic lanes/directions 
    from initial calibration frames, preventing opposing lanes from canceling each other out.
    """
    valid_list = [v for v in calibration_vectors if isinstance(v, np.ndarray) and v.shape == (2,)]
    if len(valid_list) < 10:
        return []

    valid_vecs = np.array(valid_list, dtype=np.float64)
    n_clusters = min(max_flows, len(valid_vecs))
    if n_clusters < 1:
        return []

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(valid_vecs)

    dominant_flows = []
    for center in kmeans.cluster_centers_:
        norm = np.linalg.norm(center)
        if norm > 0.3:
            dominant_flows.append(center / norm)

    return dominant_flows


def check_wrong_way_adaptive(vehicle_vector: np.ndarray, dominant_flow_vectors: list, threshold: float = -0.30) -> bool:
    """
    Evaluates vehicle motion against all learned lane clusters. If it moves against 
    all valid directions, it is flagged as a wrong-way violation.
    """
    if not dominant_flow_vectors:
        return False

    similarities = [np.dot(vehicle_vector, flow) for flow in dominant_flow_vectors]
    max_similarity = max(similarities)

    return max_similarity < threshold


# -----------------------------------------------------------------------------
# Core Frame Processing Function
# -----------------------------------------------------------------------------
def process_frame(
    frame: np.ndarray,
    frame_count: int,
    yolo_model: YOLO,
    reid_extractor: CLIPVehicleReIDExtractor,
    id_mapper: SequentialIDMapper,
    track_history: dict,
    violation_counter: dict,
    flagged_tracks: set,
    resolved_id_map: dict,
    calibration_vectors: list,
    dominant_flow_vectors: list,
    calibration_frames: int,
    video_id: str,
    violation_dir: str,
    current_fps: float = 0.0,
    last_seen_frames: dict = None,
    stabilizer: ORBStabilizer = None
) -> tuple:
    if last_seen_frames is None:
        last_seen_frames = {}

    # Step 0: Stabilize frame to neutralize camera movement/vibration
    if stabilizer is not None:
        frame = stabilizer.stabilize(frame)

    height, width = frame.shape[:2]
    detected_violations = []

    roi_polygon = np.array([
        [int(width * 0.20), int(height * 0.30)],
        [int(width * 0.70), int(height * 0.30)],
        [int(width * 0.96), int(height * 0.95)],
        [int(width * 0.04), int(height * 0.95)]
    ], np.int32)

    results = yolo_model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],  # Cars, Motorcycles, Busses, Trucks
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        currently_active_global_ids = set(resolved_id_map.values())

        for tid in track_ids:
            last_seen_frames[tid] = frame_count

        # PASS 1: Selective Re-ID Assignment
        for box, track_id in zip(boxes, track_ids):
            if track_id not in resolved_id_map:
                x1, y1, x2, y2 = map(int, box)
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(width, x2), min(height, y2)
                crop = frame[y1_c:y2_c, x1_c:x2_c]

                resolved_id = None
                if crop.size > 0 and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                    resolved_id = resolve_vehicle_identity(
                        crop=crop,
                        reid_extractor=reid_extractor,
                        frame_count=frame_count,
                        currently_active_global_ids=currently_active_global_ids,
                        similarity_threshold=0.86
                    )

                if resolved_id is None:
                    resolved_id = persistent_id_generator.mint_new_id()

                resolved_id_map[track_id] = resolved_id
                currently_active_global_ids.add(resolved_id)

        # PASS 2: Trajectory Evaluation & Adaptive Multi-Lane Wrong-Way Detection
        for box, track_id in zip(boxes, track_ids):
            if track_id not in resolved_id_map:
                continue

            x1, y1, x2, y2 = map(int, box)
            is_clipping_edge = (x1 <= 10 or y1 <= 10 or x2 >= width - 10 or y2 >= height - 10)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(width, x2), min(height, y2)
            crop = frame[y1_c:y2_c, x1_c:x2_c]

            global_id = resolved_id_map[track_id]
            display_id = id_mapper.get_seq_id(global_id)
            raw_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if global_id not in track_history:
                track_history[global_id] = []
                violation_counter[global_id] = 0

            if is_clipping_edge:
                violation_counter[global_id] = 0

            # EMA Smoothing
            if len(track_history[global_id]) > 0:
                prev_center = track_history[global_id][-1]
                smoothed_x = int(0.30 * raw_center[0] + 0.70 * prev_center[0])
                smoothed_y = int(0.30 * raw_center[1] + 0.70 * prev_center[1])
                center = (smoothed_x, smoothed_y)
            else:
                center = raw_center

            track_history[global_id].append(center)
            if len(track_history[global_id]) > 30:
                track_history[global_id].pop(0)

            is_wrong_way = False
            is_inside_roi = cv2.pointPolygonTest(roi_polygon, (float(center[0]), float(center[1])), False) >= 0

            if not is_clipping_edge and is_inside_roi and len(track_history[global_id]) >= 10:
                p_start = track_history[global_id][0]
                p_end = track_history[global_id][-1]

                dx = float(p_end[0] - p_start[0])
                dy = float(p_end[1] - p_start[1])
                dist = math.hypot(dx, dy)
                avg_speed = dist / len(track_history[global_id])

                if dist >= 130.0 and avg_speed > 2.5:
                    unit_vector = np.array([dx / dist, dy / dist], dtype=np.float64)

                    if frame_count <= calibration_frames:
                        calibration_vectors.append(unit_vector)
                    elif len(dominant_flow_vectors) > 0:
                        is_inverse_flow = check_wrong_way_adaptive(
                            unit_vector, dominant_flow_vectors, threshold=-0.30
                        )

                        if is_inverse_flow:
                            violation_counter[global_id] += 1
                        else:
                            violation_counter[global_id] = max(0, violation_counter[global_id] - 1)

                        if violation_counter[global_id] >= 5:
                            is_wrong_way = True
                            if global_id not in flagged_tracks:
                                flagged_tracks.add(global_id)

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
                    violation_counter[global_id] = 0
            else:
                violation_counter[global_id] = 0

            # Keep bounding box permanently red if the vehicle was previously flagged
            if global_id in flagged_tracks:
                is_wrong_way = True

            # Render bounding box and label
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

    # Clean up stale track maps after 30 frames
    stale_tids = [tid for tid, last_frame in last_seen_frames.items() if frame_count - last_frame > 30]
    for tid in stale_tids:
        resolved_id_map.pop(tid, None)
        last_seen_frames.pop(tid, None)

    # Draw ROI polygon boundary
    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    # Finalize calibration phase once frame limit is reached using K-Means clustering
    if frame_count == calibration_frames and len(calibration_vectors) > 0:
        flows = compute_dominant_traffic_flows(calibration_vectors)
        dominant_flow_vectors.extend(flows)

    status_str = "CALIBRATING" if frame_count <= calibration_frames else "ACTIVE"

    cv2.putText(
        frame, f"Frame: {frame_count} | Status: {status_str}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )

    cv2.putText(
        frame, f"FPS: {current_fps:.1f}",
        (width - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    return frame, detected_violations


# -----------------------------------------------------------------------------
# Video Pipeline Entry Points
# -----------------------------------------------------------------------------
def process_video_pipeline(video_id: str, input_path: str, output_path: str, violation_dir: str):
    os.makedirs(violation_dir, exist_ok=True)
    ensure_global_qdrant_collection()

    yolo_model = YOLO("yolo11n.pt")
    reid_extractor = get_clip_extractor()
    id_mapper = SequentialIDMapper()
    stabilizer = ORBStabilizer()

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
    resolved_id_map = {}
    last_seen_frames = {}
    detected_violations = []

    calibration_vectors = []
    dominant_flow_vectors = []
    calibration_frames = 90
    frame_count = 0

    prev_time = time.time()
    current_fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        time_diff = now - prev_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
        prev_time = now

        frame_count += 1

        frame, violations = process_frame(
            frame, frame_count, yolo_model, reid_extractor, id_mapper, track_history,
            violation_counter, flagged_tracks, resolved_id_map, calibration_vectors,
            dominant_flow_vectors, calibration_frames, video_id, violation_dir,
            current_fps=current_fps, last_seen_frames=last_seen_frames, stabilizer=stabilizer
        )
        detected_violations.extend(violations)
        out.write(frame)

    cap.release()
    out.release()
    return detected_violations


def stream_video_pipeline(video_id: str, input_path: str, violation_dir: str):
    os.makedirs(violation_dir, exist_ok=True)
    ensure_global_qdrant_collection()

    yolo_model = YOLO("yolo11n.pt")
    reid_extractor = get_clip_extractor()
    id_mapper = SequentialIDMapper()
    stabilizer = ORBStabilizer()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {input_path}")

    track_history = {}
    violation_counter = {}
    flagged_tracks = set()
    resolved_id_map = {}
    last_seen_frames = {}

    calibration_vectors = []
    dominant_flow_vectors = []
    calibration_frames = 90
    frame_count = 0

    prev_time = time.time()
    current_fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            time_diff = now - prev_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
            prev_time = now

            frame_count += 1
            frame, _ = process_frame(
                frame, frame_count, yolo_model, reid_extractor, id_mapper, track_history,
                violation_counter, flagged_tracks, resolved_id_map, calibration_vectors,
                dominant_flow_vectors, calibration_frames, video_id, violation_dir,
                current_fps=current_fps, last_seen_frames=last_seen_frames, stabilizer=stabilizer
            )

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()