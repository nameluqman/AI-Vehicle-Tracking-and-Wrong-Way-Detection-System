import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchreid
import onnxruntime as ort
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# -------------------------------------------------------------------
# Automated Re-ID Model Exporter
# -------------------------------------------------------------------
class FeatureExtractorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def export_osnet_to_onnx(onnx_path: str):
    """Generates and exports the OSNet Re-ID model to ONNX format if missing."""
    print(f"[Export] ONNX model missing at '{onnx_path}'. Generating now...")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    base_model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        loss="softmax",
        pretrained=True
    )

    base_model.eval()
    if hasattr(base_model, "classifier"):
        base_model.classifier = nn.Identity()

    wrapper = FeatureExtractorWrapper(base_model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[Export] Success! Model generated ({file_size_mb:.2f} MB)")


def ensure_onnx_model_exists(onnx_path: str):
    if not os.path.exists(onnx_path):
        export_osnet_to_onnx(onnx_path)


def ensure_custom_bytetrack_config(config_path: str = "custom_bytetrack.yaml"):
    """Ensures custom ByteTrack YAML with optimized parameters exists."""
    content = """tracker_type: bytetrack
track_high_thresh: 0.15
track_low_thresh: 0.05
new_track_thresh: 0.15
track_buffer: 90
match_thresh: 0.8
fuse_score: True
"""
    with open(config_path, "w") as f:
        f.write(content)


# -------------------------------------------------------------------
# Configuration & Model Initialization
# -------------------------------------------------------------------
ONNX_MODEL_PATH = os.path.join("models", "vehicle-reid-0001", "osnet_x1_0_vehicle_reid.onnx")
ensure_onnx_model_exists(ONNX_MODEL_PATH)
ensure_custom_bytetrack_config("custom_bytetrack.yaml")

available_providers = ort.get_available_providers()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Switched to yolo11m.pt for better detection accuracy on small objects
yolo_model = YOLO("yolo11m.pt")

# Qdrant Database Setup
QDRANT_DB_PATH = os.path.join(os.getcwd(), "qdrant_db")
os.makedirs(QDRANT_DB_PATH, exist_ok=True)

qdrant_client = QdrantClient(path=QDRANT_DB_PATH)
COLLECTION_NAME = "vehicle_reid"

try:
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
except Exception as e:
    print(f"[Qdrant] Error initializing collection: {e}")

GLOBAL_VEHICLE_COUNTER = 0


# -------------------------------------------------------------------
# Re-ID Vector & Database Operations
# -------------------------------------------------------------------
def pad_to_square(crop: np.ndarray) -> np.ndarray:
    h, w, c = crop.shape
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim, c), dtype=crop.dtype)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = crop
    return padded


def preprocess_crop_onnx(crop: np.ndarray) -> np.ndarray:
    padded = pad_to_square(crop)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))

    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def extract_embedding(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return np.zeros(512, dtype=np.float32)

    tensor = preprocess_crop_onnx(crop)
    outputs = ort_session.run([output_name], {input_name: tensor})
    embedding = outputs[0].squeeze()

    if embedding.ndim > 1:
        embedding = embedding.flatten()[:512]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def get_or_assign_sequential_id(embedding: np.ndarray, similarity_threshold: float = 0.95) -> int:
    global GLOBAL_VEHICLE_COUNTER

    if np.all(embedding == 0) or np.isnan(embedding).any():
        GLOBAL_VEHICLE_COUNTER += 1
        return GLOBAL_VEHICLE_COUNTER

    try:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding.tolist(),
            limit=1,
            with_vectors=False,
        )

        search_result = response.points

        if search_result and search_result[0].score >= similarity_threshold:
            return search_result[0].payload["sequential_id"]

        GLOBAL_VEHICLE_COUNTER += 1
        new_id = GLOBAL_VEHICLE_COUNTER

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=new_id,
                    vector=embedding.tolist(),
                    payload={"sequential_id": new_id},
                )
            ],
        )
        return new_id
    except Exception as e:
        print(f"[ReID Error] Assigning ID: {e}")
        GLOBAL_VEHICLE_COUNTER += 1
        return GLOBAL_VEHICLE_COUNTER


# -------------------------------------------------------------------
# Calibration & Road Assignment Helpers
# -------------------------------------------------------------------
def extract_calibration_points(calib_info: dict):
    lines = calib_info.get("lines", {})
    up_pts = calib_info.get("upward_points") or lines.get("upward") or [(330, 360), (660, 360)]
    down_pts = calib_info.get("downward_points") or lines.get("downward") or [(690, 360), (1240, 360)]
    return up_pts, down_pts


def draw_calibration_lines(frame: np.ndarray, calib_info: dict) -> np.ndarray:
    up_pts, down_pts = extract_calibration_points(calib_info)

    if up_pts and len(up_pts) == 2:
        pt1 = tuple(map(int, up_pts[0]))
        pt2 = tuple(map(int, up_pts[1]))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
        cv2.putText(
            frame,
            "UPWARD ROAD (ROAD 1)",
            (pt1[0], max(30, pt1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if down_pts and len(down_pts) == 2:
        pt1 = tuple(map(int, down_pts[0]))
        pt2 = tuple(map(int, down_pts[1]))
        cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
        cv2.putText(
            frame,
            "DOWNWARD ROAD (ROAD 2)",
            (pt1[0], max(30, pt1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    return frame


def get_road_center_x(calib_info: dict):
    up_pts, down_pts = extract_calibration_points(calib_info)
    
    road_up_x = (up_pts[0][0] + up_pts[1][0]) / 2.0 if len(up_pts) == 2 else 495.0
    road_down_x = (down_pts[0][0] + down_pts[1][0]) / 2.0 if len(down_pts) == 2 else 965.0
    
    return road_up_x, road_down_x


def is_vehicle_wrong_way(
    centroid: np.ndarray,
    positions_history: list,
    calib_info: dict,
    min_trajectory_len: int = 5,
    displacement_threshold: int = 15,
) -> bool:
    if len(positions_history) < min_trajectory_len:
        return False

    p_start = positions_history[0]
    p_end = positions_history[-1]

    delta_y = p_end[1] - p_start[1]

    if abs(delta_y) < displacement_threshold:
        return False

    road_up_x, road_down_x = get_road_center_x(calib_info)
    cx = centroid[0]

    dist_to_up = abs(cx - road_up_x)
    dist_to_down = abs(cx - road_down_x)

    if dist_to_up < dist_to_down:
        if delta_y > displacement_threshold:
            return True
    else:
        if delta_y < -displacement_threshold:
            return True

    return False


# -------------------------------------------------------------------
# Real-Time Generator Streaming Pipeline
# -------------------------------------------------------------------
def stream_video_pipeline(
    video_id: str,
    input_path: str,
    violation_dir: str,
    calib_info: dict,
    resize_dim: tuple = (1280, 720),
    enable_reid: bool = True,
    violation_callback=None,
):
    os.makedirs(violation_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    positions = {}
    violation_counter = {}
    saved_violations = set()
    local_track_to_seq_id = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = cv2.resize(frame, resize_dim)
        draw_calibration_lines(frame, calib_info)

        results = yolo_model.track(
            frame,
            persist=True,
            tracker="custom_bytetrack.yaml",
            classes=vehicle_classes,
            conf=0.15,        # Low confidence threshold for small vehicles
            imgsz=1280,       # Native resolution inference (avoids downscaling small objects)
            iou=0.30,         # Lower IoU for fast-moving distant objects
            verbose=False,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()

            for box, obj_id in zip(boxes, track_ids):
                x3, y3, x4, y4 = box
                obj_id = int(obj_id)
                centroid = np.array([int((x3 + x4) / 2.0), int((y3 + y4) / 2.0)])

                if obj_id not in positions:
                    positions[obj_id] = []
                    violation_counter[obj_id] = 0

                    if enable_reid:
                        x3_i, y3_i, x4_i, y4_i = map(int, [x3, y3, x4, y4])
                        crop = frame[
                            max(0, y3_i):min(frame.shape[0], y4_i),
                            max(0, x3_i):min(frame.shape[1], x4_i),
                        ]
                        if crop.size > 0:
                            embedding = extract_embedding(crop)
                            sequential_id = get_or_assign_sequential_id(embedding, similarity_threshold=0.95)
                            local_track_to_seq_id[obj_id] = sequential_id
                        else:
                            local_track_to_seq_id[obj_id] = obj_id

                positions[obj_id].append(centroid)
                if len(positions[obj_id]) > 15:
                    positions[obj_id].pop(0)

                is_wrong_way = is_vehicle_wrong_way(
                    centroid,
                    positions[obj_id],
                    calib_info,
                    min_trajectory_len=5,
                    displacement_threshold=15,
                )

                if is_wrong_way:
                    violation_counter[obj_id] += 1
                else:
                    violation_counter[obj_id] = max(0, violation_counter[obj_id] - 1)

                x3_i, y3_i, x4_i, y4_i = map(int, [x3, y3, x4, y4])
                display_id = local_track_to_seq_id.get(obj_id, obj_id)

                if violation_counter[obj_id] >= 8:
                    color = (0, 0, 255)
                    label = f"WRONG WAY: ID {display_id}"

                    if obj_id not in saved_violations:
                        crop_filename = f"violation_{display_id}_{frame_idx}.jpg"
                        crop_path = os.path.join(violation_dir, crop_filename)

                        crop = frame[
                            max(0, y3_i):min(frame.shape[0], y4_i),
                            max(0, x3_i):min(frame.shape[1], x4_i),
                        ]
                        if crop.size > 0 and cv2.imwrite(crop_path, crop):
                            violation_record = {
                                "id": f"viol_{video_id}_{display_id}_{frame_idx}",
                                "video_id": video_id,
                                "track_id": display_id,
                                "filename": crop_filename,
                                "image_url": f"/media/violations_{video_id}/{crop_filename}",
                                "timestamp_frame": frame_idx,
                                "violation_type": "Wrong Way Driving",
                            }

                            saved_violations.add(obj_id)
                            if violation_callback:
                                violation_callback(violation_record)

                elif violation_counter[obj_id] > 0:
                    color = (0, 165, 255)
                    label = f"Monitoring: ID {display_id}"
                else:
                    color = (0, 255, 0)
                    label = f"ID {display_id}"

                cv2.rectangle(frame, (x3_i, y3_i), (x4_i, y4_i), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x3_i, max(20, y3_i - 10)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    color,
                    2,
                )

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


# -------------------------------------------------------------------
# Offline Video Processing Pipeline
# -------------------------------------------------------------------
def process_video_pipeline(
    video_id: str,
    input_path: str,
    output_path: str,
    violation_dir: str,
    calib_info: dict,
    displacement_threshold: int = 15,
    frame_persistence: int = 8,
    resize_dim: tuple = (1280, 720),
    enable_reid: bool = True,
) -> list:
    os.makedirs(violation_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {input_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_path, fourcc, input_fps, resize_dim)

    vehicle_classes = [2, 3, 5, 7]
    positions = {}
    violation_counter = {}
    saved_violations = set()
    detected_violations_list = []
    local_track_to_seq_id = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = cv2.resize(frame, resize_dim)
        draw_calibration_lines(frame, calib_info)

        results = yolo_model.track(
            frame,
            persist=True,
            tracker="custom_bytetrack.yaml",
            classes=vehicle_classes,
            conf=0.15,        # Low confidence threshold for small vehicles
            imgsz=1280,       # Native resolution inference
            iou=0.30,         # Lower IoU for fast-moving distant objects
            verbose=False,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()

            for box, obj_id in zip(boxes, track_ids):
                x3, y3, x4, y4 = box
                obj_id = int(obj_id)
                centroid = np.array([int((x3 + x4) / 2.0), int((y3 + y4) / 2.0)])

                if obj_id not in positions:
                    positions[obj_id] = []
                    violation_counter[obj_id] = 0

                    if enable_reid:
                        x3_i, y3_i, x4_i, y4_i = map(int, [x3, y3, x4, y4])
                        crop = frame[
                            max(0, y3_i):min(frame.shape[0], y4_i),
                            max(0, x3_i):min(frame.shape[1], x4_i),
                        ]
                        if crop.size > 0:
                            embedding = extract_embedding(crop)
                            sequential_id = get_or_assign_sequential_id(embedding, similarity_threshold=0.95)
                            local_track_to_seq_id[obj_id] = sequential_id
                        else:
                            local_track_to_seq_id[obj_id] = obj_id

                positions[obj_id].append(centroid)
                if len(positions[obj_id]) > 15:
                    positions[obj_id].pop(0)

                is_wrong_way = is_vehicle_wrong_way(
                    centroid,
                    positions[obj_id],
                    calib_info,
                    min_trajectory_len=5,
                    displacement_threshold=displacement_threshold,
                )

                if is_wrong_way:
                    violation_counter[obj_id] += 1
                else:
                    violation_counter[obj_id] = max(0, violation_counter[obj_id] - 1)

                x3_i, y3_i, x4_i, y4_i = map(int, [x3, y3, x4, y4])
                display_id = local_track_to_seq_id.get(obj_id, obj_id)

                if violation_counter[obj_id] >= frame_persistence:
                    color = (0, 0, 255)
                    label = f"WRONG WAY: ID {display_id}"

                    if obj_id not in saved_violations:
                        crop_filename = f"violation_{display_id}_{frame_idx}.jpg"
                        crop_path = os.path.join(violation_dir, crop_filename)

                        crop = frame[
                            max(0, y3_i):min(frame.shape[0], y4_i),
                            max(0, x3_i):min(frame.shape[1], x4_i),
                        ]
                        if crop.size > 0 and cv2.imwrite(crop_path, crop):
                            violation_record = {
                                "id": f"viol_{video_id}_{display_id}_{frame_idx}",
                                "video_id": video_id,
                                "track_id": display_id,
                                "filename": crop_filename,
                                "image_url": f"/media/violations_{video_id}/{crop_filename}",
                                "timestamp_frame": frame_idx,
                                "violation_type": "Wrong Way Driving",
                            }

                            detected_violations_list.append(violation_record)
                            saved_violations.add(obj_id)
                elif violation_counter[obj_id] > 0:
                    color = (0, 165, 255)
                    label = f"Monitoring: ID {display_id}"
                else:
                    color = (0, 255, 0)
                    label = f"ID {display_id}"

                cv2.rectangle(frame, (x3_i, y3_i), (x4_i, y4_i), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x3_i, max(20, y3_i - 10)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    color,
                    2,
                )

        out_video.write(frame)

    cap.release()
    out_video.release()
    return detected_violations_list