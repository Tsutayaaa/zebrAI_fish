import cv2
import numpy as np
import os
import time
import queue
import threading
import logging
import matplotlib.pyplot as plt
from math import sqrt
import sys
from ultralytics import YOLO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 常量定义
TEXT_POS_SPEED = (10, 30)
TEXT_POS_FREEZE = (10, 60)
TEXT_POS_TOP = (10, 90)
TEXT_POS_DISPLACEMENT = (10, 120)
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2
MAX_TRACK_POINTS = 1000

# 默认配置参数
CONFIG = {
    "paths": {
        "video_path": "C:/Users/he/OneDrive/Documents/YOLOv12/yolov12-main/_DemoTest/video/Big_Fish.mp4",
        "model_path": "./zebrafish_AI_v1.3.pt",
        "output_base_path": "",
        "video_output_name": "Untitled",
        "use_camera": False,
        "camera_index": 0,
    },
    "features": {
        "save_preview_video": False,
        "calculate_top_time": True,
        "calculate_speed": True,
        "calculate_total_displacement": True,
        "calculate_freeze_time": True,
        "generate_heatmap": True,
        "generate_trajectory": True,
        "generate_timeline": True,
        "generate_speed_plot": True,
    },
    "preview_settings": {
        "preview_heatmap": False,
        "preview_trajectory": True,
        "trajectory_length": 10,
        "preview_scale": 0.8,
        "preview_fps": 10
    },
    "tank": {
        "tank_shape": "rectangle",
        "scale_factor": 5,
        "trapezoid": {"real_width_top_mm": 270, "real_width_bottom_mm": 220, "real_height_mm": 145},
        "rectangle": {"real_width_mm": 200, "real_height_mm": 200},
    },
    "image_processing": {
        "heatmap_kernel_size": (61, 61),
        "heatmap_alpha": 0.4,
        "frame_alpha": 0.6,
        "background_color": (0, 0, 0),
        "add_filename_to_image": True,
        "font_scale": FONT_SCALE,
        "font_color": FONT_COLOR,
        "font_thickness": FONT_THICKNESS,
        "resize_factor": 0.6
    },
    "freeze_detection": {
        "freeze_speed_threshold": 10.0,
        "freeze_displacement_threshold": 2.0,
        "freeze_wh_change_threshold": 5.0,
        "freeze_confidence_threshold": 0.8,
        "freeze_duration_threshold": 1.5,
        "top_duration_threshold": 1.5,
        "window_size": 5,
    },
    "recording": {
        "max_record_time": 300,
        "skip_frames": 1,
        "save_raw_video": True,
    }
}

class VideoProcessor:
    video_sequence = 1

    def __init__(self, config):
        self.config = config
        self._validate_config()
        self._initialize_dimensions()
        self.points = []
        self.frame = None
        self.selected_point = None
        self.perspective_matrix = None
        self.cap = self._open_video()
        self.output_dir = None
        self.division_line_y = None
        self.resize_factor = self.config["image_processing"].get("resize_factor", 1)
        self.tank_shape = self.config["tank"].get("tank_shape", "rectangle")

    def _validate_config(self):
        required_keys = [("paths", "model_path"), ("tank", "scale_factor"), ("recording", "max_record_time")]
        for section, key in required_keys:
            if section not in self.config or key not in self.config[section]:
                raise ValueError(f"Missing required config key: {section}.{key}")
        if not self.config["paths"]["use_camera"] and not self.config["paths"]["video_path"]:
            raise ValueError("Video path must be provided if not using camera")

    def _initialize_dimensions(self):
        scale = self.config["tank"]["scale_factor"]
        resize_factor = self.config["image_processing"].get("resize_factor", 1)
        if self.config["tank"]["tank_shape"] == "trapezoid":
            self.dst_width_top = int(self.config["tank"]["trapezoid"]["real_width_top_mm"] * scale * resize_factor)
            self.dst_width_bottom = int(self.config["tank"]["trapezoid"]["real_width_bottom_mm"] * scale * resize_factor)
            self.dst_height = int(self.config["tank"]["trapezoid"]["real_height_mm"] * scale * resize_factor)
        else:
            self.dst_width = int(self.config["tank"]["rectangle"]["real_width_mm"] * scale * resize_factor)
            self.dst_height = int(self.config["tank"]["rectangle"]["real_height_mm"] * scale * resize_factor)

    def _open_video(self):
        if self.config["paths"]["use_camera"]:
            cap = cv2.VideoCapture(self.config["paths"]["camera_index"])
        else:
            cap = cv2.VideoCapture(self.config["paths"]["video_path"])
        if not cap.isOpened():
            logging.error("Error: Could not open video source")
            return None
        return cap

    def _select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                logging.info(f"Point added: ({x}, {y}), Selected: {len(self.points)}")
            else:
                for i, pt in enumerate(self.points):
                    if abs(pt[0] - x) < 10 and abs(pt[1] - y) < 10:
                        self.selected_point = i
                        logging.info(f"Selected point {i} for adjustment: ({x}, {y})")
                        break
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
            self.points[self.selected_point] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point = None

    def _draw_selection_with_preview(self):
        if self.frame is None:
            logging.error("No frame available for drawing selection")
            return
        temp_frame = self.frame.copy()
        for i, pt in enumerate(self.points):
            cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(temp_frame, self.points[i - 1], pt, (0, 255, 0), 2)
        cv2.putText(temp_frame, "Select 4 corners of the tank in any order", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        if len(self.points) < 4:
            cv2.putText(temp_frame, f"Points selected: {len(self.points)}/4", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        else:
            cv2.line(temp_frame, self.points[3], self.points[0], (0, 255, 0), 2)
            if self.tank_shape == "trapezoid":
                cv2.putText(temp_frame, "Press Enter to confirm, then L/R for side", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            else:
                cv2.putText(temp_frame, "Press Enter to confirm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.imshow("Select Corners", temp_frame)

    def _sort_points(self):
        if len(self.points) != 4:
            return self.points
        pts = np.array(self.points, dtype=np.float32)
        sorted_y = sorted(pts, key=lambda p: p[1])
        top_pts = sorted_y[:2]
        bottom_pts = sorted_y[2:]
        top_pts = sorted(top_pts, key=lambda p: p[0])
        bottom_pts = sorted(bottom_pts, key=lambda p: p[0])
        return [
            (float(top_pts[0][0]), float(top_pts[0][1])),
            (float(top_pts[1][0]), float(top_pts[1][1])),
            (float(bottom_pts[1][0]), float(bottom_pts[1][1])),
            (float(bottom_pts[0][0]), float(bottom_pts[0][1]))
        ]

    def _select_trapezoid_side(self):
        logging.info("Press 'L' for left trapezoid side or 'R' for right trapezoid side")
        while True:
            if self.frame is None:
                logging.error("No frame available for trapezoid side selection")
                raise ValueError("Frame is None during trapezoid side selection")
            temp_frame = self.frame.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(temp_frame, self.points[i - 1], pt, (0, 255, 0), 2)
            cv2.line(temp_frame, self.points[3], self.points[0], (0, 255, 0), 2)
            cv2.putText(temp_frame, "Press L for left side or R for right side", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            cv2.imshow("Select Corners", temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('l'), ord('L')]:
                return "left"
            elif key in [ord('r'), ord('R')]:
                return "right"
            elif key == ord('q'):
                raise KeyboardInterrupt("User aborted side selection")

    def _get_dst_points_trapezoid(self, trapezoid_side):
        if trapezoid_side == "right":
            return np.float32([
                [0, 0], [self.dst_width_top, 0],
                [self.dst_width_bottom, self.dst_height],
                [0, self.dst_height]
            ])
        else:
            return np.float32([
                [0, 0], [self.dst_width_top, 0],
                [self.dst_width_top, self.dst_height],
                [self.dst_width_top - self.dst_width_bottom, self.dst_height]
            ])

    def _get_dst_points_rectangle(self):
        return np.float32([
            [0, 0], [self.dst_width, 0],
            [self.dst_width, self.dst_height],
            [0, self.dst_height]
        ])

    def _select_division_line(self):
        if not self.config["features"]["calculate_top_time"]:
            return
        temp_frame = self.frame.copy()
        if self.tank_shape == "trapezoid":
            warped_frame = cv2.warpPerspective(temp_frame, self.perspective_matrix,
                                               (self.dst_width_top, self.dst_height))
            max_height = self.dst_height
            max_width = self.dst_width_top
        else:
            warped_frame = cv2.warpPerspective(temp_frame, self.perspective_matrix,
                                               (self.dst_width, self.dst_height))
            max_height = self.dst_height
            max_width = self.dst_width
        self.division_line_y = max_height // 2
        dragging = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal dragging
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                self.division_line_y = max(0, min(y, max_height - 1))
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        cv2.namedWindow("Set Division Line")
        cv2.setMouseCallback("Set Division Line", mouse_callback)
        logging.info("Drag to set the horizontal division line, press Enter to confirm")

        while True:
            temp_frame_display = warped_frame.copy()
            cv2.line(temp_frame_display, (0, self.division_line_y), (max_width, self.division_line_y), (0, 0, 255), 2)
            cv2.putText(temp_frame_display, "Drag to set division line, press Enter to confirm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            cv2.imshow("Set Division Line", temp_frame_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
            elif key == ord('q'):
                raise KeyboardInterrupt("User aborted division line selection")

        cv2.destroyWindow("Set Division Line")
        logging.info(f"Division line set at y={self.division_line_y}")
        return self.division_line_y

    def init_perspective_transform(self):
        if not self.cap:
            logging.error("Video capture not initialized")
            return None, None
        ret, self.frame = self.cap.read()
        if not ret:
            logging.error("Failed to read first frame")
            raise ValueError("Error: Could not read first frame")

        self.frame = cv2.resize(self.frame, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Select Corners")
        cv2.setMouseCallback("Select Corners", self._select_points)
        logging.info("Please select 4 corners of the fish tank in any order")

        try:
            while True:
                self._draw_selection_with_preview()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("User aborted corner selection")
                    raise KeyboardInterrupt("User aborted selection")
                elif len(self.points) == 4 and key == 13:
                    break
        except Exception as e:
            logging.error(f"Error during corner selection: {e}")
            cv2.destroyWindow("Select Corners")
            raise

        if len(self.points) != 4:
            logging.error(f"Invalid number of points selected: {len(self.points)}")
            cv2.destroyWindow("Select Corners")
            raise ValueError("Error: Exactly 4 points must be selected")

        sorted_points = self._sort_points()
        src_points = np.float32(sorted_points)

        if self.tank_shape == "trapezoid":
            try:
                trapezoid_side = self._select_trapezoid_side()
                dst_points = self._get_dst_points_trapezoid(trapezoid_side)
                self.config["tank"]["trapezoid_side"] = trapezoid_side
            except Exception as e:
                logging.error(f"Error during trapezoid side selection: {e}")
                cv2.destroyWindow("Select Corners")
                raise
        else:
            dst_points = self._get_dst_points_rectangle()

        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        if self.config["features"]["calculate_top_time"]:
            try:
                self._select_division_line()
            except Exception as e:
                logging.error(f"Error during division line selection: {e}")
                raise

        folder_name = f"{self.config['paths']['video_output_name']}_{VideoProcessor.video_sequence:03d}"
        self.output_dir = os.path.join(self.config["paths"]["output_base_path"], folder_name)
        os.makedirs(self.output_dir, exist_ok=True)

        cv2.destroyWindow("Select Corners")
        logging.info("Perspective transform initialized successfully")
        return self.perspective_matrix, self.output_dir

    def process_frame(self, frame):
        if self.resize_factor != 1:
            frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor, interpolation=cv2.INTER_AREA)
        if self.tank_shape == "trapezoid":
            return cv2.warpPerspective(frame, self.perspective_matrix, (self.dst_width_top, self.dst_height))
        return cv2.warpPerspective(frame, self.perspective_matrix, (self.dst_width, self.dst_height))

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("VideoProcessor resources cleaned up")

class TrackerAnalyzer:
    def __init__(self, config, video_processor):
        self.config = config
        self.processor = video_processor
        self.model = self._load_model()
        self.output_dir = None
        self.perspective_matrix = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.video_writer = None
        self.writer_thread = None
        self.total_displacement_mm = 0.0

    def _load_model(self):
        model_path = self.config["paths"]["model_path"]
        logging.info(f"Attempting to load model from path: {model_path}")
        if not os.path.exists(model_path):
            logging.warning(f"Configured model path does not exist: {model_path}")
            default_path = "./zebrafish_AI_v1.3.pt"
            logging.info(f"Checking default path: {default_path}")
            if os.path.exists(default_path):
                model_path = default_path
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                alt_path = os.path.join(base_dir, "zebrafish_AI_v1.3.pt")
                logging.info(f"Checking runtime directory: {alt_path}")
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    if hasattr(sys, '_MEIPASS'):
                        meipass_path = os.path.join(sys._MEIPASS, "zebrafish_AI_v1.3.pt")
                        logging.info(f"Checking PyInstaller temp path: {meipass_path}")
                        if os.path.exists(meipass_path):
                            model_path = meipass_path
                        else:
                            logging.error(f"All local paths invalid: {self.config['paths']['model_path']}")
                            return None
                    else:
                        logging.error(f"All local paths invalid: {self.config['paths']['model_path']}")
                        return None
        try:
            model = YOLO(model_path, task='detect')
            model.model_path = model_path
            return model
        except Exception as e:
            logging.error(f"Failed to load YOLO model - {e}")
            return None

    def _analyze_behavior(self, tracked_data, track_points, motion_window, speeds_mm_s, timeline,
                          top_time, top_times, top_start, top_duration, freeze_time, freeze_times,
                          freeze_start, current_freeze_duration, display_freeze_time, was_in_top):
        if len(tracked_data) <= 1:
            return (top_time, top_times, top_start, top_duration, freeze_time, freeze_times,
                    freeze_start, current_freeze_duration, display_freeze_time, was_in_top)

        prev_x, prev_y, prev_w, prev_h, prev_t = tracked_data[-2]
        curr_x, curr_y, curr_w, curr_h, curr_t = tracked_data[-1]
        pixel_distance = sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        distance_mm = pixel_distance / self.config["tank"]["scale_factor"]
        w_change = abs(curr_w - prev_w)
        h_change = abs(curr_h - prev_h)
        time_diff = curr_t - prev_t

        if time_diff <= 0:
            return (top_time, top_times, top_start, top_duration, freeze_time, freeze_times,
                    freeze_start, current_freeze_duration, display_freeze_time, was_in_top)

        speed_mm_s = distance_mm / time_diff
        motion_window.append((speed_mm_s, distance_mm, w_change, h_change, time_diff))
        if len(motion_window) > self.config["freeze_detection"]["window_size"]:
            motion_window.pop(0)

        avg_speed = np.mean([item[0] for item in motion_window])
        avg_displacement = np.mean([item[1] for item in motion_window])
        avg_w_change = np.mean([item[2] for item in motion_window])
        avg_h_change = np.mean([item[3] for item in motion_window])

        speed_confidence = 1 - min(avg_speed / self.config["freeze_detection"]["freeze_speed_threshold"], 1)
        displacement_confidence = 1 - min(
            avg_displacement / self.config["freeze_detection"]["freeze_displacement_threshold"], 1)
        wh_confidence = 1 - min(
            max(avg_w_change, avg_h_change) / self.config["freeze_detection"]["freeze_wh_change_threshold"], 1)
        freeze_confidence = (0.4 * speed_confidence + 0.4 * displacement_confidence + 0.2 * wh_confidence)
        if self.config["features"]["calculate_freeze_time"] and freeze_confidence > self.config["freeze_detection"][
            "freeze_confidence_threshold"]:
            current_freeze_duration += time_diff
            display_freeze_time += time_diff
            if freeze_start is None:
                freeze_start = prev_t
            speed_mm_s = 0
        else:
            if freeze_start is not None and current_freeze_duration >= self.config["freeze_detection"][
                "freeze_duration_threshold"]:
                freeze_times.append((freeze_start, curr_t))
                freeze_time += current_freeze_duration
            freeze_start = None
            current_freeze_duration = 0
            speed_mm_s = min(avg_speed, 1000)

        speeds_mm_s.append(speed_mm_s)

        if self.config["features"]["calculate_top_time"] and self.processor.division_line_y is not None:
            center_y = curr_y
            if center_y < self.processor.division_line_y:
                if not was_in_top:
                    top_start = prev_t
                    top_duration = 0
                    was_in_top = True
                top_duration += time_diff
                top_time += time_diff
                timeline.append(1)
            else:
                if was_in_top and top_duration >= self.config["freeze_detection"]["top_duration_threshold"]:
                    top_times.append((top_start, curr_t))
                top_start = None
                top_duration = 0
                was_in_top = False
                timeline.append(0)
        else:
            if was_in_top and top_duration >= self.config["freeze_detection"]["top_duration_threshold"]:
                top_times.append((top_start, curr_t))
            top_start = None
            top_duration = 0
            was_in_top = False
            timeline.append(0)

        if self.config["features"]["calculate_total_displacement"]:
            self.total_displacement_mm += distance_mm

        return (top_time, top_times, top_start, top_duration, freeze_time, freeze_times,
                freeze_start, current_freeze_duration, display_freeze_time, was_in_top)

    def _update_preview_frame(self, warped_frame, track_points, boxes):
        preview_frame = warped_frame.copy()
        if self.config["preview_settings"]["preview_trajectory"] and self.config["features"]["generate_trajectory"]:
            max_length = self.config["preview_settings"]["trajectory_length"]
            limited_track_points = track_points[-max_length:] if len(track_points) > max_length else track_points
            preview_frame = self._draw_trajectory(preview_frame, limited_track_points)
        if len(boxes) > 0:
            x, y, w, h = boxes[0]
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(preview_frame, top_left, bottom_right, (0, 255, 0), 2)
        return preview_frame

    def _draw_stats(self, frame, speeds_mm_s, display_freeze_time, top_time, elapsed_time):
        y_offset = 30
        if self.config["features"]["calculate_speed"] and len(speeds_mm_s) > 0:
            speed_text = f"Speed: {speeds_mm_s[-1]:.1f} mm/s"
            cv2.putText(frame, speed_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            y_offset += 30
        if self.config["features"]["calculate_freeze_time"]:
            freeze_text = f"Freeze Time: {display_freeze_time:.2f} s"
            cv2.putText(frame, freeze_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            y_offset += 30
        if self.config["features"]["calculate_top_time"]:
            top_text = f"Top Time: {top_time:.2f} s"
            cv2.putText(frame, top_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            y_offset += 30
        if self.config["features"]["calculate_total_displacement"]:
            displacement_text = f"Total Displacement: {self.total_displacement_mm:.1f} mm"
            cv2.putText(frame, displacement_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
            y_offset += 30
        elapsed_time_text = f"Elapsed: {elapsed_time:.1f} s"
        text_size = cv2.getTextSize(elapsed_time_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        cv2.putText(frame, elapsed_time_text, (text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    def _prepare_output_paths(self):
        heatmap_path = os.path.join(self.output_dir, "heatmap_last_frame.png") if self.config["features"]["generate_heatmap"] else None
        track_path = os.path.join(self.output_dir, "track_last_frame.png") if self.config["features"]["generate_trajectory"] else None
        video_name = "camera" if self.config["paths"]["use_camera"] else os.path.splitext(os.path.basename(self.config["paths"]["video_path"]))[0]
        heatmap_data_path = os.path.join(self.output_dir,
                                         f"{self.config['paths']['video_output_name']}_{VideoProcessor.video_sequence:03d}_heatmap_data.npy") if self.config["features"]["generate_heatmap"] else None
        behavior_data_path = os.path.join(self.output_dir,
                                          f"{self.config['paths']['video_output_name']}_{VideoProcessor.video_sequence:03d}_behavior_data.npy")
        timeline_path = os.path.join(self.output_dir, "timeline_chart.png") if self.config["features"]["generate_timeline"] else None
        speed_plot_path = os.path.join(self.output_dir, "speed_plot.png") if self.config["features"]["generate_speed_plot"] else None
        video_filename = f"{self.config['paths']['video_output_name']}_{VideoProcessor.video_sequence:03d}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        raw_video_path = os.path.join(self.output_dir, f"{self.config['paths']['video_output_name']}_{VideoProcessor.video_sequence:03d}_raw.mp4")
        return heatmap_path, track_path, video_name, heatmap_data_path, behavior_data_path, timeline_path, speed_plot_path, video_path, raw_video_path

    def _initialize_video_writer(self, fps, video_path):
        if not self.config["features"]["save_preview_video"]:
            return None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.config["tank"]["tank_shape"] == "trapezoid":
            width = self.processor.dst_width_top
            height = self.processor.dst_height
        else:
            width = self.processor.dst_width
            height = self.processor.dst_height
        return cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    def _video_writer_thread(self, stop_event, video_path, nominal_fps):
        if stop_event is None:
            stop_event = threading.Event()
        frames = []
        start_time = time.perf_counter()
        while not stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                frames.append(frame)
            except queue.Empty:
                continue
        end_time = time.perf_counter()

        frame_count = len(frames)
        effective_fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else nominal_fps
        logging.info(f"Video writer: {frame_count} frames written in {end_time - start_time:.2f} seconds, effective FPS: {effective_fps:.2f}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.config["tank"]["tank_shape"] == "trapezoid":
            width = self.processor.dst_width_top
            height = self.processor.dst_height
        else:
            width = self.processor.dst_width
            height = self.processor.dst_height
        writer = cv2.VideoWriter(video_path, fourcc, effective_fps, (width, height))

        for frame in frames:
            writer.write(frame)
        writer.release()
        logging.info("Video writer thread terminated")

    def _write_raw_video(self, raw_frames, raw_video_path, nominal_fps):
        if not raw_frames:
            logging.error("No raw frames to write.")
            return

        frame_count = len(raw_frames)
        first_timestamp = raw_frames[0][1]
        last_timestamp = raw_frames[-1][1]
        recording_duration = last_timestamp - first_timestamp

        effective_fps = (frame_count - 1) / recording_duration if recording_duration > 0 else nominal_fps
        logging.info(f"Writing raw video: {frame_count} frames over {recording_duration:.2f} seconds. Effective FPS: {effective_fps:.2f}")

        height, width = raw_frames[0][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(raw_video_path, fourcc, effective_fps, (width, height))

        for frame, _ in raw_frames:
            writer.write(frame)
        writer.release()
        logging.info(f"Raw video saved to: {raw_video_path}")

    def _finalize_behavior_analysis(self, track_points, last_frame, total_time,
                                    top_time, top_times, top_start, top_duration,
                                    freeze_time, freeze_times, freeze_start,
                                    current_freeze_duration, was_in_top,
                                    speeds_mm_s, timeline):
        if freeze_start is not None and current_freeze_duration >= self.config["freeze_detection"]["freeze_duration_threshold"]:
            freeze_times.append((freeze_start, total_time))
            freeze_time += current_freeze_duration
        if was_in_top and top_duration >= self.config["freeze_detection"]["top_duration_threshold"]:
            top_times.append((top_start, total_time))
        logging.info(f"Total Frames Written: {len(track_points)}")
        logging.info(f"Total Freeze Time: {freeze_time:.2f} seconds, Frequency: {len(freeze_times)}")
        logging.info(f"Total Top Time: {top_time:.2f} seconds, Frequency: {len(top_times)}")
        logging.info(f"Total Recorded Time: {total_time:.2f} seconds")
        logging.info(f"Total Displacement: {self.total_displacement_mm:.1f} mm")

    def track_and_analyze(self, stop_event=None, frame_queue=None):
        if not self.model or not self.processor.cap:
            logging.error("Model or video capture not initialized")
            return

        fps = self.processor.cap.get(cv2.CAP_PROP_FPS) if not self.config["paths"]["use_camera"] else 30
        frame_time = 1.0 / fps
        max_record_time = self.config["recording"]["max_record_time"]
        buffer_duration = 1.0 if self.config["paths"]["use_camera"] else 2.0
        buffer_frames = int(buffer_duration * fps)
        skip_frames = self.config["recording"]["skip_frames"]

        logging.info(f"Tracking started with max_record_time: {max_record_time} seconds, FPS: {fps}, Buffer frames: {buffer_frames}")

        try:
            self.perspective_matrix, self.output_dir = self.processor.init_perspective_transform()
        except (ValueError, KeyboardInterrupt) as e:
            logging.error(str(e))
            self.processor.cleanup()
            return

        if self.config["paths"]["use_camera"]:
            logging.info("Buffering camera stream...")
            start_buffer = time.perf_counter()
            while time.perf_counter() - start_buffer < buffer_duration:
                self.processor.cap.read()
        else:
            logging.info("Buffering video stream...")
            for _ in range(buffer_frames):
                ret, _ = self.processor.cap.read()
                if not ret:
                    logging.warning("Video ended during buffering")
                    self.processor.cleanup()
                    return

        recording_start = time.perf_counter()
        logging.info("Buffer complete, starting detection")

        (heatmap_path, track_path, video_name, heatmap_data_path, behavior_data_path,
         timeline_path, speed_plot_path, video_path, raw_video_path) = self._prepare_output_paths()

        if self.config["features"]["save_preview_video"]:
            self.writer_thread = threading.Thread(target=self._video_writer_thread, args=(stop_event, video_path, fps),
                                                  daemon=True)
            self.writer_thread.start()

        raw_frames = []
        if self.config["recording"].get("save_raw_video", False) and self.config["paths"]["use_camera"]:
            logging.info("Recording raw video to buffer...")

        track_points = []
        tracked_data = []
        frame_count = 0
        top_time = 0.0
        top_times = []
        top_start = None
        top_duration = 0
        freeze_time = 0.0
        freeze_times = []
        freeze_start = None
        current_freeze_duration = 0
        display_freeze_time = 0
        timeline = []
        was_in_top = False
        speeds_mm_s = []
        motion_window = []
        last_frame = None

        if self.config["tank"]["tank_shape"] == "trapezoid":
            background = np.full((self.processor.dst_height, self.processor.dst_width_top, 3),
                                 self.config["image_processing"]["background_color"], dtype=np.uint8)
        else:
            background = np.full((self.processor.dst_height, self.processor.dst_width, 3),
                                 self.config["image_processing"]["background_color"], dtype=np.uint8)
        mask = self._create_tank_mask()

        try:
            while True:
                if stop_event and stop_event.is_set():
                    logging.info("Tracking stopped by UI or 'q'.")
                    break

                frame_start = time.perf_counter()
                ret, frame = self.processor.cap.read()
                if not ret:
                    logging.info("End of video reached.")
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue

                if self.config["recording"].get("save_raw_video", False) and self.config["paths"]["use_camera"]:
                    raw_frames.append((frame.copy(), time.perf_counter()))

                elapsed_time = time.perf_counter() - recording_start
                if elapsed_time >= max_record_time:
                    logging.info(f"Reached max record time of {max_record_time} seconds.")
                    break

                warped_frame = self.processor.process_frame(frame)
                boxes = self.model(warped_frame)[0].boxes.xywh.cpu().numpy()
                logging.debug(f"Frame {frame_count}: Detected {len(boxes)} objects")

                if len(boxes) > 0:
                    x, y, w, h = boxes[0]
                    center_x, center_y = int(x), int(y)
                    w_mm = w / self.config["tank"]["scale_factor"]
                    h_mm = h / self.config["tank"]["scale_factor"]
                    track_points.append((center_x, center_y))
                    tracked_data.append((center_x, center_y, w_mm, h_mm, elapsed_time))
                    (top_time, top_times, top_start, top_duration, freeze_time, freeze_times, freeze_start,
                     current_freeze_duration, display_freeze_time, was_in_top) = self._analyze_behavior(
                        tracked_data, track_points, motion_window, speeds_mm_s, timeline,
                        top_time, top_times, top_start, top_duration, freeze_time, freeze_times,
                        freeze_start, current_freeze_duration, display_freeze_time, was_in_top)

                output_frame = background.copy()
                output_frame[mask == 255] = warped_frame[mask == 255]
                if self.config["features"]["calculate_top_time"] and self.processor.division_line_y is not None:
                    cv2.line(output_frame, (0, self.processor.division_line_y),
                             (output_frame.shape[1], self.processor.division_line_y), (0, 0, 255), 2)

                if track_points:
                    preview_frame = self._update_preview_frame(warped_frame, track_points, boxes)
                    output_preview = background.copy()
                    output_preview[mask == 255] = preview_frame[mask == 255]
                    if self.config["features"]["calculate_top_time"] and self.processor.division_line_y is not None:
                        cv2.line(output_preview, (0, self.processor.division_line_y),
                                 (output_preview.shape[1], self.processor.division_line_y), (0, 0, 255), 2)
                    self._draw_stats(output_preview, speeds_mm_s, display_freeze_time, top_time, elapsed_time)
                    if self.config["features"]["save_preview_video"]:
                        try:
                            self.frame_queue.put(output_preview, timeout=0.1)
                        except queue.Full:
                            pass
                    if frame_queue:
                        try:
                            frame_queue.put(output_preview, timeout=0.1)
                        except queue.Full:
                            pass
                    last_frame = warped_frame
                else:
                    if self.config["features"]["save_preview_video"]:
                        try:
                            self.frame_queue.put(output_frame, timeout=0.1)
                        except queue.Full:
                            pass
                    if frame_queue:
                        try:
                            frame_queue.put(output_frame, timeout=0.1)
                        except queue.Full:
                            pass
                    last_frame = warped_frame

                process_time = time.perf_counter() - frame_start
                wait_time = frame_time - process_time
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    logging.debug(f"Frame {frame_count} processing too slow: {process_time:.3f}s > {frame_time:.3f}s")

            recording_end = time.perf_counter()
            recording_duration = recording_end - recording_start
            logging.info(f"Total recording duration: {recording_duration:.2f} seconds")

            if track_points and last_frame is not None:
                self._finalize_behavior_analysis(track_points, last_frame, elapsed_time,
                                                 top_time, top_times, top_start, top_duration,
                                                 freeze_time, freeze_times, freeze_start,
                                                 current_freeze_duration, was_in_top,
                                                 speeds_mm_s, timeline)
                self._save_results(track_points, last_frame, background, mask, video_name,
                                   heatmap_path, track_path, heatmap_data_path, behavior_data_path, timeline,
                                   top_time, len(top_times), elapsed_time, fps, tracked_data,
                                   speeds_mm_s, timeline_path, speed_plot_path,
                                   freeze_time, len(freeze_times), freeze_times, top_times)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            self.processor.cleanup()
            if self.config["features"]["save_preview_video"]:
                if stop_event:
                    stop_event.set()
                if self.writer_thread:
                    self.writer_thread.join(timeout=1.0)
            if raw_frames:
                self._write_raw_video(raw_frames, raw_video_path, fps)

    def _generate_heatmap(self, frame, track_points):
        if self.config["tank"]["tank_shape"] == "trapezoid":
            heatmap_raw = np.zeros((self.processor.dst_height, self.processor.dst_width_top), dtype=np.float32)
        else:
            heatmap_raw = np.zeros((self.processor.dst_height, self.processor.dst_width), dtype=np.float32)
        for point in track_points:
            if 0 <= point[1] < heatmap_raw.shape[0] and 0 <= point[0] < heatmap_raw.shape[1]:
                heatmap_raw[point[1], point[0]] += 1
        heatmap_smoothed = cv2.GaussianBlur(heatmap_raw, self.config["image_processing"]["heatmap_kernel_size"], 0)
        heatmap_normalized = cv2.normalize(heatmap_smoothed, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)
        heatmap_visual = cv2.addWeighted(frame, self.config["image_processing"]["frame_alpha"],
                                         heatmap_color, self.config["image_processing"]["heatmap_alpha"], 0)
        return heatmap_visual, heatmap_raw

    def _draw_trajectory(self, frame, track_points):
        track_frame = frame.copy()
        for i in range(1, len(track_points)):
            alpha = i / len(track_points)
            color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
            cv2.line(track_frame, track_points[i - 1], track_points[i], color, 1)
        return track_frame

    def _create_tank_mask(self):
        if self.config["tank"]["tank_shape"] == "trapezoid":
            mask = np.zeros((self.processor.dst_height, self.processor.dst_width_top), dtype=np.uint8)
            if self.config["tank"].get("trapezoid_side") == "right":
                pts = np.array([[0, 0], [self.processor.dst_width_top, 0],
                                [self.processor.dst_width_bottom, self.processor.dst_height],
                                [0, self.processor.dst_height]], dtype=np.int32)
            else:
                pts = np.array([[0, 0], [self.processor.dst_width_top, 0],
                                [self.processor.dst_width_top, self.processor.dst_height],
                                [self.processor.dst_width_top - self.processor.dst_width_bottom,
                                 self.processor.dst_height]], dtype=np.int32)
        else:
            mask = np.zeros((self.processor.dst_height, self.processor.dst_width), dtype=np.uint8)
            pts = np.array([[0, 0], [self.processor.dst_width, 0],
                            [self.processor.dst_width, self.processor.dst_height],
                            [0, self.processor.dst_height]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _add_filename_to_image(self, image, filename):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(filename, font, self.config["image_processing"]["font_scale"],
                                    self.config["image_processing"]["font_thickness"])[0]
        text_x = image.shape[1] - text_size[0] - 10
        text_y = image.shape[0] - 10
        cv2.putText(image, filename, (text_x, text_y), font,
                    self.config["image_processing"]["font_scale"],
                    self.config["image_processing"]["font_color"],
                    self.config["image_processing"]["font_thickness"])
        return image

    def _save_behavior_data(self, filepath, speeds_mm_s, top_time, top_frequency, freeze_time, freeze_times, top_times,
                            freeze_frequency):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {}

        data_to_save["tank_shape"] = self.config["tank"]["tank_shape"]
        data_to_save["trapezoid_side"] = self.config["tank"].get("trapezoid_side", None)
        data_to_save["scale_factor"] = self.config["tank"]["scale_factor"]

        if self.config["features"]["calculate_total_displacement"]:
            data_to_save["total_displacement_mm"] = self.total_displacement_mm

        if self.config["features"]["calculate_speed"]:
            avg_speed = np.mean(speeds_mm_s) if speeds_mm_s else 0
            data_to_save["avg_speed_mm_s"] = avg_speed
            data_to_save["speeds_mm_s"] = speeds_mm_s

        if self.config["features"]["calculate_top_time"]:
            data_to_save["top_time"] = float(top_time)
            data_to_save["top_times"] = top_times
            data_to_save["top_frequency"] = top_frequency

        if self.config["features"]["calculate_freeze_time"]:
            data_to_save["freeze_time"] = float(freeze_time)
            data_to_save["freeze_times"] = freeze_times
            data_to_save["freeze_frequency"] = freeze_frequency

        np.save(filepath, data_to_save, allow_pickle=True)
        logging.info(f"Behavior data saved to: {filepath}")

    def _save_heatmap_data(self, filepath, heatmap_data):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {
            "tank_shape": self.config["tank"]["tank_shape"],
            "trapezoid_side": self.config["tank"].get("trapezoid_side", None),
            "scale_factor": self.config["tank"]["scale_factor"],
            "heatmap_data": heatmap_data
        }
        np.save(filepath, data_to_save, allow_pickle=True)
        logging.info(f"Heatmap data saved to: {filepath}")

    def _save_stats(self, top_times, freeze_times, total_time, video_name, top_time, top_frequency, freeze_time,
                    freeze_frequency, output_path):
        plt.figure(figsize=(15, 4))
        for i, (start, end) in enumerate(top_times):
            label = f"Top time ({top_time:.2f}s, {top_frequency})" if i == 0 else None
            plt.barh(y=1, width=end - start, left=start, height=0.4, color='red', label=label)
        for i, (start, end) in enumerate(freeze_times):
            label = f"Freeze time ({freeze_time:.2f}s, {freeze_frequency})" if i == 0 else None
            plt.barh(y=0, width=end - start, left=start, height=0.4, color='blue', label=label)
        plt.xlabel("Time (s)")
        plt.title(f"Time Line: {video_name}")
        plt.yticks([0, 1], ["Freeze Time", "Top Time"])
        plt.xlim(0, total_time)
        plt.ylim(-0.5, 1.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Timeline chart saved to: {output_path}")

    def _save_speed_plot(self, tracked_data, speeds_mm_s, total_time, video_name, output_path):
        times = [data[4] for data in tracked_data[1:]]
        plt.figure(figsize=(15, 5))
        avg_speed = np.mean(speeds_mm_s) if speeds_mm_s else 0
        plt.plot(times, speeds_mm_s, label=f"Speed (avg:{avg_speed:.1f} mm/s)", color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (mm/s)")
        plt.title(f"Speed: {video_name}")
        plt.xlim(0, total_time)
        plt.ylim(0, max(speeds_mm_s + [1]) * 1.2)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Speed plot saved to: {output_path}")

    def _save_results(self, track_points, warped_frame, background, mask, video_name,
                      heatmap_path, track_path, heatmap_data_path, behavior_data_path, timeline,
                      top_time, top_frequency, total_time, fps, tracked_data,
                      speeds_mm_s, timeline_path, speed_plot_path,
                      freeze_time, freeze_frequency, freeze_times, top_times):
        self._save_behavior_data(behavior_data_path, speeds_mm_s, top_time, top_frequency, freeze_time, freeze_times,
                                 top_times, freeze_frequency)

        # 修改热图保存逻辑
        if self.config["features"]["generate_heatmap"]:
            if track_points and warped_frame is not None:
                overlay_heatmap, heatmap_raw = self._generate_heatmap(warped_frame, track_points)
                heatmap_output = background.copy()
                heatmap_output[mask == 255] = overlay_heatmap[mask == 255]
                if self.config["image_processing"]["add_filename_to_image"]:
                    heatmap_output = self._add_filename_to_image(heatmap_output, video_name)
                cv2.imwrite(heatmap_path, heatmap_output)
                logging.info(f"Heatmap image saved to: {heatmap_path}")
            else:
                # 如果没有 track_points 或 warped_frame，生成空热图
                if self.config["tank"]["tank_shape"] == "trapezoid":
                    heatmap_raw = np.zeros((self.processor.dst_height, self.processor.dst_width_top), dtype=np.float32)
                else:
                    heatmap_raw = np.zeros((self.processor.dst_height, self.processor.dst_width), dtype=np.float32)
                logging.info("No valid track points or frame, saving empty heatmap.")
            self._save_heatmap_data(heatmap_data_path, heatmap_raw)

        # 以下保持不变
        if self.config["features"]["generate_trajectory"] and track_points:
            track_output = background.copy()
            track_output[mask == 255] = self._draw_trajectory(warped_frame, track_points)[mask == 255]
            if self.config["image_processing"]["add_filename_to_image"]:
                track_output = self._add_filename_to_image(track_output, video_name)
            cv2.imwrite(track_path, track_output)
            logging.info(f"Track image saved to: {track_path}")

        if self.config["features"]["generate_timeline"] and (
                self.config["features"]["calculate_top_time"] or self.config["features"]["calculate_freeze_time"]):
            self._save_stats(top_times, freeze_times, total_time, video_name, top_time, top_frequency, freeze_time,
                             freeze_frequency, timeline_path)

        if self.config["features"]["generate_speed_plot"] and self.config["features"][
            "calculate_speed"] and speeds_mm_s:
            self._save_speed_plot(tracked_data, speeds_mm_s, total_time, video_name, speed_plot_path)

        if self.config["features"]["calculate_total_displacement"]:
            logging.info(f"Total Displacement saved: {self.total_displacement_mm:.1f} mm")
        if self.config["features"]["calculate_speed"]:
            avg_speed = np.mean(speeds_mm_s) if speeds_mm_s else 0
            logging.info(f"Average Speed saved: {avg_speed:.1f} mm/s")

def main():
    processor = VideoProcessor(CONFIG)
    tracker = TrackerAnalyzer(CONFIG, processor)
    tracker.track_and_analyze()

if __name__ == "__main__":
    main()