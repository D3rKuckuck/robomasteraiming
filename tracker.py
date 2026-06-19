import math
import time

import cv2
import torch
from ultralytics import YOLO


def resolve_device(requested: str) -> str:
    """Возвращает 'cuda' если запрошено GPU и CUDA доступна, иначе 'cpu'."""
    if requested == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class PersonTracker:
    TRACK_TIMEOUT = 2.0
    SEARCH_TIMEOUT = 5.0
    SEARCH_ROTATION_SPEED = 1.0  # рад/с вокруг оси при поиске

    def __init__(self, model_path='yolo11n.pt', device='cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.tracks = {}           # {track_id: (center_x, center_y)}
        self.selected_id = None
        self.last_seen_time = {}   # {track_id: timestamp}
        self.is_searching = False
        self.search_start_time = None
        self.search_direction = 1  # 1 = вправо, -1 = влево

    def process_frame(self, frame):
        """Запускает детекцию+трекинг, рисует боксы, обновляет self.tracks."""
        results = self.model.track(
            frame,
            persist=True,
            tracker="custom_track.yaml",
            classes=[0],
            conf=0.3,
            verbose=True,
            device=self.device,
        )

        current_time = time.time()
        current_tracks = {}

        if results[0].boxes.id is not None:
            boxes       = results[0].boxes.xyxy.cpu().numpy()
            track_ids   = results[0].boxes.id.int().cpu().tolist()
            class_ids   = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                if cls_id != 0 or conf <= 0.3:
                    continue
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                current_tracks[track_id] = (cx, cy)
                self.last_seen_time[track_id] = current_time

                is_target = (track_id == self.selected_id)
                color     = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 3 if is_target else 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.circle(frame, (int(cx), int(cy)), 6, color, -1)
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        self.tracks = current_tracks
        return frame

    def select_nearest(self, x, y, radius=100):
        """Выбирает ближайший трек к точке (x, y) в координатах кадра."""
        best_id   = None
        best_dist = float('inf')
        for track_id, (cx, cy) in self.tracks.items():
            dist = math.hypot(cx - x, cy - y)
            if dist < best_dist and dist < radius:
                best_dist = dist
                best_id   = track_id

        if best_id is not None:
            self.selected_id = best_id
            self.is_searching = False
            self.last_seen_time[best_id] = time.time()
        return best_id

    def reset(self):
        self.selected_id = None
        self.is_searching = False
        self.search_start_time = None

    def is_target_lost(self):
        """True, если выбранная цель не появлялась дольше TRACK_TIMEOUT."""
        if self.selected_id is None:
            return False
        if self.selected_id in self.tracks:
            self.last_seen_time[self.selected_id] = time.time()
            return False
        last = self.last_seen_time.get(self.selected_id)
        if last is None:
            return True
        return time.time() - last > self.TRACK_TIMEOUT

    def begin_search(self):
        if not self.is_searching:
            self.is_searching = True
            self.search_start_time = time.time()
            self.search_direction = 1

    def search_timed_out(self):
        if not self.is_searching or self.search_start_time is None:
            return False
        return time.time() - self.search_start_time > self.SEARCH_TIMEOUT

    def search_time_left(self):
        if not self.is_searching or self.search_start_time is None:
            return 0.0
        return max(0.0, self.SEARCH_TIMEOUT - (time.time() - self.search_start_time))
