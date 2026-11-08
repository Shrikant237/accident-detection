
import cv2
import time
import math
import csv
from collections import defaultdict, deque
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "C:/Users/77200/yolov5/accident.mp4"

CONFIDENCE = 0.35

# COCO classes:
# 2 = Car, 3 = Motorcycle, 5 = Bus, 7 = Truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Pixel-based speed thresholds
MOVING_THRESHOLD = 25.0
STOPPED_THRESHOLD = 8.0

# Collision detection
IOU_THRESHOLD = 0.15
PROLONGED_COLLISION_FRAMES = 10

# Sudden stop detection
SUDDEN_STOP_WINDOW = 2.0

# Event logging
LOG_FILE = "accident_log.csv"


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def calculate_speed(previous, current, time_difference):

    if time_difference <= 0:
        return 0

    distance = math.dist(previous, current)

    return distance / time_difference


def get_timestamp(seconds):

    return time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime()
    )


# ==========================================
# ACCIDENT LOGGING
# ==========================================

def log_accident(event, vehicle_ids, video_time):

    timestamp = get_timestamp(video_time)

    print(
        f"[ALERT] {event} | "
        f"Vehicles: {vehicle_ids} | "
        f"Video time: {video_time:.2f}s"
    )

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            round(video_time, 2),
            event,
            str(vehicle_ids)
        ])


# ==========================================
# MAIN ACCIDENT DETECTION
# ==========================================

def detect_accident(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    frame_count = 0

    # Vehicle tracking history
    previous_centers = {}
    previous_speeds = {}

    # Last time a vehicle was moving
    last_motion_time = {}

    # Collision duration for each vehicle pair
    collision_frames = defaultdict(int)

    # Avoid repeated accident logs
    reported_accidents = set()

    # Create CSV file with headers
    with open(LOG_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "System Timestamp",
            "Video Timestamp",
            "Event",
            "Vehicle IDs"
        ])

    print("Starting YOLOv8 Accident Detection...")
    print("Press Q to exit.")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        video_time = frame_count / fps

        # ==================================
        # YOLO DETECTION + BYTETRACK
        # ==================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        current_vehicles = {}

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, cls, confidence in zip(
                boxes, ids, classes, confidences
            ):

                center = calculate_center(box)

                current_vehicles[track_id] = {
                    "box": box,
                    "center": center,
                    "class": cls,
                    "confidence": confidence
                }

        # ==================================
        # VEHICLE SPEED CALCULATION
        # ==================================

        current_speeds = {}

        for track_id, vehicle in current_vehicles.items():

            center = vehicle["center"]

            speed = 0.0

            if track_id in previous_centers:

                previous_center = previous_centers[track_id]

                speed = calculate_speed(
                    previous_center,
                    center,
                    1.0 / fps
                )

            current_speeds[track_id] = speed

            if speed >= MOVING_THRESHOLD:

                last_motion_time[track_id] = video_time

            # ==================================
            # SUDDEN STOP DETECTION
            # ==================================

            sudden_stop = False

            if track_id in previous_speeds:

                old_speed = previous_speeds[track_id]

                recently_moving = (
                    video_time - last_motion_time.get(
                        track_id, -100
                    ) <= SUDDEN_STOP_WINDOW
                )

                if (
                    old_speed >= MOVING_THRESHOLD
                    and speed <= STOPPED_THRESHOLD
                    and recently_moving
                ):

                    sudden_stop = True

                    event_key = ("sudden_stop", track_id)

                    if event_key not in reported_accidents:

                        log_accident(
                            "Sudden Stop - Possible Accident",
                            [track_id],
                            video_time
                        )

                        reported_accidents.add(event_key)

            # ==================================
            # DRAW VEHICLE INFORMATION
            # ==================================

            x1, y1, x2, y2 = map(int, vehicle["box"])

            color = (0, 255, 0)

            if sudden_stop:
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            class_name = model.names[vehicle["class"]]

            label = (
                f"{class_name} ID:{track_id} "
                f"Speed:{speed:.1f}px/s"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            if sudden_stop:

                cv2.putText(
                    frame,
                    "SUDDEN STOP!",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # ==================================
        # VEHICLE COLLISION DETECTION
        # ==================================

        current_pairs = set()

        vehicle_ids = list(current_vehicles.keys())

        for i in range(len(vehicle_ids)):

            for j in range(i + 1, len(vehicle_ids)):

                id1 = vehicle_ids[i]
                id2 = vehicle_ids[j]

                box1 = current_vehicles[id1]["box"]
                box2 = current_vehicles[id2]["box"]

                iou = calculate_iou(box1, box2)

                pair = tuple(sorted((id1, id2)))

                if iou >= IOU_THRESHOLD:

                    current_pairs.add(pair)

                    collision_frames[pair] += 1

                    # Check whether collision persists
                    if (
                        collision_frames[pair]
                        >= PROLONGED_COLLISION_FRAMES
                    ):

                        # Require evidence of recent movement
                        # followed by both vehicles slowing down.
                        recently_moving = any(
                            video_time - last_motion_time.get(
                                vid, -100
                            ) <= SUDDEN_STOP_WINDOW
                            for vid in pair
                        )

                        both_stopped = all(
                            current_speeds.get(vid, 0)
                            <= STOPPED_THRESHOLD
                            for vid in pair
                        )

                        if recently_moving and both_stopped:

                            event_key = ("collision", pair)

                            if event_key not in reported_accidents:

                                log_accident(
                                    "Possible Vehicle Collision",
                                    pair,
                                    video_time
                                )

                                reported_accidents.add(event_key)

                else:

                    collision_frames[pair] = 0

        # Reset pairs that are no longer overlapping
        for pair in list(collision_frames.keys()):

            if pair not in current_pairs:
                collision_frames[pair] = 0

        # ==================================
        # DISPLAY INFORMATION
        # ==================================

        cv2.putText(
            frame,
            f"Vehicles Detected: {len(current_vehicles)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Display possible collision warning
        active_collisions = [
            pair for pair, count in collision_frames.items()
            if count >= PROLONGED_COLLISION_FRAMES
        ]

        if active_collisions:

            cv2.putText(
                frame,
                "POSSIBLE ACCIDENT DETECTED!",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "ROAD STATUS: NORMAL",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "YOLOv8 Road Accident Detection",
            frame
        )

        # Update tracking history
        previous_centers = {
            vid: vehicle["center"]
            for vid, vehicle in current_vehicles.items()
        }

        previous_speeds = current_speeds.copy()

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nAccident detection completed.")
    print(f"Accident log saved to: {LOG_FILE}")


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    detect_accident(VIDEO_PATH)
