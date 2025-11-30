import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
from main_tracker import VideoProcessor, TrackerAnalyzer, CONFIG
import logging
import os
import cv2
import multiprocessing as mp
import queue

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 语言配置（参数说明保持原样）
LANGUAGES = {
    "English": {
        "title": "zebrAI_fish (1.1.0_beta)",
        "main_interface": "Main Interface",
        "tank_settings": "Tank Settings",
        "scale_factor": "Scale Factor (px/mm)",
        "trapezoid_width_top": "Trapezoid Top Width (mm)",
        "trapezoid_width_bottom": "Trapezoid Bottom Width (mm)",
        "trapezoid_height": "Trapezoid Height (mm)",
        "rectangle_width": "Rectangle Width (mm)",
        "rectangle_height": "Rectangle Height (mm)",
        "image_processing": "Image Processing",
        "heatmap_kernel_size": "Heatmap Kernel Size (w,h)",
        "heatmap_alpha": "Heatmap Alpha",
        "frame_alpha": "Frame Alpha",
        "font_scale": "Font Scale",
        "font_thickness": "Font Thickness",
        "resize_factor": "Resize Factor",
        "freeze_detection": "Freeze/Top Detection",
        "freeze_speed_threshold": "Freeze Speed Threshold (mm/s)",
        "freeze_displacement_threshold": "Freeze Displacement Threshold (mm)",
        "freeze_wh_change_threshold": "Freeze WH Change Threshold (mm)",
        "freeze_confidence_threshold": "Freeze Confidence Threshold",
        "freeze_duration_threshold": "Freeze Duration Threshold (s)",
        "top_duration_threshold": "Top Duration Threshold (s)",
        "window_size": "Window Size",
        "input_source": "Input Source",
        "video_file": "Video File",
        "camera": "Camera",
        "browse": "Browse",
        "camera_index": "Camera Device",
        "tank_type": "Tank Type",
        "trapezoid": "Trapezoid(NTT)",
        "rectangle": "Rectangle",
        "options_output": "Options & Output",
        "save_preview_video": "Save Preview Video",
        "calculate_top_time": "Calculate Top Time",
        "calculate_speed": "Calculate Speed",
        "calculate_freeze_time": "Calculate Freeze Time",
        "generate_heatmap": "Generate Heatmap",
        "generate_trajectory": "Generate Trajectory",
        "generate_timeline": "Generate Timeline",
        "generate_speed_plot": "Generate Speed Plot",
        "calculate_total_displacement": "Record Total Displacement",
        "preview_settings": "Preview Settings",
        "preview_heatmap": "Preview Heatmap",
        "preview_trajectory": "Preview Trajectory",
        "trajectory_length": "Trajectory Length",
        "output_path": "Output Path",
        "group_name": "Group Name",
        "group_name_placeholder": "e.g.: Control",
        "sequence_number": "Number",
        "model": "Model: {}",
        "log_output": "Log",
        "start_tracking": "Start Tracking",
        "stop_tracking": "Stop Tracking",
        "record_time": "Record Time",
        "set_record_time": "Set Record Time",
        "minutes": "Minutes",
        "seconds": "Seconds",
        "skip_frames": "Skip Frames",
        "export_preview": "Export Preview",
        "language_selection": "Language",
        "error_invalid": "Invalid {}",
        "error_missing": "Please select a video file or use camera",
    },
    "Chinese": {
        "title": "zebrAI_fish (1.1.0_beta)",
        "main_interface": "主界面",
        "tank_settings": "鱼缸参数",
        "scale_factor": "缩放因子 (px/mm)",
        "trapezoid_width_top": "梯形顶部宽度 (mm)",
        "trapezoid_width_bottom": "梯形底部宽度 (mm)",
        "trapezoid_height": "梯形高度 (mm)",
        "rectangle_width": "矩形宽度 (mm)",
        "rectangle_height": "矩形高度 (mm)",
        "image_processing": "图像处理",
        "heatmap_kernel_size": "热图核大小 (w,h)",
        "heatmap_alpha": "热图透明度",
        "frame_alpha": "帧透明度",
        "font_scale": "字体大小",
        "font_thickness": "字体粗细",
        "resize_factor": "分辨率缩放因子",
        "freeze_detection": "Freeze/Top设置",
        "freeze_speed_threshold": "冻结速度阈值 (mm/s)",
        "freeze_displacement_threshold": "Freeze位移阈值 (mm)",
        "freeze_wh_change_threshold": "Freeze宽高变化阈值 (mm)",
        "freeze_confidence_threshold": "Freeze置信度阈值",
        "freeze_duration_threshold": "Freeze持续时间阈值 (s)",
        "top_duration_threshold": "Top持续时间阈值 (s)",
        "window_size": "窗口大小",
        "input_source": "输入源",
        "video_file": "视频文件",
        "camera": "摄像头",
        "browse": "浏览",
        "camera_index": "摄像头设备",
        "tank_type": "鱼缸类型",
        "trapezoid": "梯形(NTT)",
        "rectangle": "矩形",
        "options_output": "选项和输出",
        "save_raw_video": "保存原视频",
        "save_preview_video": "保存预览视频",
        "calculate_top_time": "计算顶部时间",
        "calculate_speed": "计算速度",
        "calculate_freeze_time": "计算Freeze时间",
        "generate_heatmap": "生成热力图",
        "generate_trajectory": "生成轨迹图",
        "generate_timeline": "生成时间轴图",
        "generate_speed_plot": "生成速度图",
        "calculate_total_displacement": "记录总位移",
        "preview_settings": "预览设置",
        "preview_heatmap": "预览热图",
        "preview_trajectory": "预览轨迹",
        "trajectory_length": "轨迹长度",
        "output_path": "输出路径",
        "group_name": "组名称",
        "group_name_placeholder": "例如：Control",
        "sequence_number": "编号",
        "model": "模型: {}",
        "log_output": "日志",
        "start_tracking": "开始跟踪",
        "stop_tracking": "停止跟踪",
        "record_time": "记录时间",
        "set_record_time": "设置录制时长",
        "minutes": "分",
        "seconds": "秒",
        "skip_frames": "跳帧数",
        "export_preview": "导出预览",
        "language_selection": "语言",
        "error_invalid": "无效的 {}",
        "error_missing": "请选择视频文件或使用摄像头",
    }
}

# 参数解释保持原有注释，不影响实时检测与计时
PARAM_INFO = {
    "scale_factor": {"English": "Pixels per millimeter for scaling physical measurements.", "Chinese": "用于缩放物理测量的每毫米像素数。"},
    "trapezoid_width_top": {"English": "Top width of the trapezoid tank in millimeters.", "Chinese": "梯形鱼缸顶部宽度（毫米）。"},
    "trapezoid_width_bottom": {"English": "Bottom width of the trapezoid tank in millimeters.", "Chinese": "梯形鱼缸底部宽度（毫米）。"},
    "trapezoid_height": {"English": "Height of the trapezoid tank in millimeters.", "Chinese": "梯形鱼缸高度（毫米）。"},
    "rectangle_width": {"English": "Width of the rectangle tank in millimeters.", "Chinese": "矩形鱼缸宽度（毫米）。"},
    "rectangle_height": {"English": "Height of the rectangle tank in millimeters.", "Chinese": "矩形鱼缸高度（毫米）。"},
    "heatmap_kernel_size": {"English": "Size of Gaussian kernel for heatmap smoothing (width, height).", "Chinese": "热图平滑的高斯核大小（宽，高）。"},
    "heatmap_alpha": {"English": "Transparency of heatmap overlay (0.0 to 1.0).", "Chinese": "热图叠加的透明度（0.0 到 1.0）。"},
    "frame_alpha": {"English": "Transparency of the original frame (0.0 to 1.0).", "Chinese": "原始帧的透明度（0.0 到 1.0）。"},
    "font_scale": {"English": "Font size for text overlays.", "Chinese": "文本叠加的字体大小。"},
    "font_thickness": {"English": "Thickness of text overlays.", "Chinese": "文本叠加的粗细。"},
    "resize_factor": {"English": "Factor to resize input frames (0.1-1.0, smaller values improve performance).", "Chinese": "输入帧的缩放因子（0.1-1.0，较小值提升性能）。"},
    "freeze_speed_threshold": {"English": "Maximum speed (mm/s) to consider as frozen.", "Chinese": "视为冻结的最大速度（毫米/秒）。"},
    "freeze_displacement_threshold": {"English": "Maximum displacement (mm) to consider as frozen.", "Chinese": "视为冻结的最大位移（毫米）。"},
    "freeze_wh_change_threshold": {"English": "Maximum width/height change (mm) to consider as frozen.", "Chinese": "视为冻结的最大宽高变化（毫米）。"},
    "freeze_confidence_threshold": {"English": "Minimum confidence (0-1) to detect freeze.", "Chinese": "检测冻结的最小置信度（0-1）。"},
    "freeze_duration_threshold": {"English": "Minimum duration (s) to count as a freeze event.", "Chinese": "计为冻结事件的最小持续时间（秒）。"},
    "top_duration_threshold": {"English": "Minimum duration (s) to count as a top event.", "Chinese": "计为顶部事件的最小持续时间（秒）。"},
    "window_size": {"English": "Number of frames for smoothing freeze detection.", "Chinese": "平滑冻结检测的帧数。"},
    "group_name": {"English": "Base name for exported video files (e.g., 'Control').", "Chinese": "导出视频文件的基础名称（例如 'Control'）。"},
    "sequence_number": {"English": "Number for video files, auto-incremented after each recording.", "Chinese": "视频文件的编号，每次录制后自动递增。"},
    "preview_heatmap": {"English": "Show heatmap in real-time preview (disable for performance).", "Chinese": "在实时预览中显示热图（建议关闭以优化性能）。"},
    "preview_trajectory": {"English": "Show trajectory in real-time preview.", "Chinese": "在实时预览中显示轨迹。"},
    "trajectory_length": {"English": "Maximum number of points to display in trajectory preview.", "Chinese": "轨迹预览中显示的最大点数。"},
    "preview_resolution": {"English": "Resolution for the preview window (width,height).", "Chinese": "预览窗口分辨率（宽,高）。"},
    "preview_fps": {"English": "Frame rate for preview video.", "Chinese": "预览视频帧率。"},
    "skip_frames": {"English": "Number of frames to skip between processing (1 = no skip).", "Chinese": "处理之间跳过的帧数（1=不跳过）。"},
    "calculate_total_displacement": {"English": "Record the total displacement of the tracked object (mm).", "Chinese": "记录物体的总位移（毫米）。"}
}

def display_process(frame_queue, stop_event):
    """独立的进程，用于显示 OpenCV 预览窗口"""
    # 这里直接使用从队列中获取的帧，不再缩放
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            if frame is not None:
                cv2.imshow('Track & Heatmap Preview', frame)
            key = cv2.waitKey(30) & 0xFF  # 固定延时，保持响应
            if key == ord('q'):
                stop_event.set()
        except queue.Empty:
            continue
    cv2.destroyAllWindows()
    logging.info("Display process terminated")

class TrackerUI:
    def __init__(self, root):
        self.root = root
        self.config = CONFIG.copy()
        self.tracker = None
        self.processor = None
        self.running_thread = None
        self.display_process = None
        self.frame_queue = mp.Queue(maxsize=10)
        self.stop_event = mp.Event()
        self.running = False
        self.language = "English"
        self.camera_devices = [(f"Camera {i}", i) for i in range(4)]
        self.sequence_number_var = tk.IntVar(value=1)
        self.tooltip = None
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.title(LANGUAGES[self.language]["title"])

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))

        self.main_interface_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_interface_frame, text=LANGUAGES[self.language]["main_interface"])

        # 输入源选择
        self.input_frame = ttk.LabelFrame(self.main_interface_frame, text=LANGUAGES[self.language]["input_source"],
                                          padding="5")
        self.input_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        self.input_var = tk.StringVar(value="Video File")
        ttk.Radiobutton(self.input_frame, text=LANGUAGES[self.language]["video_file"],
                        variable=self.input_var, value="Video File", command=self.toggle_input_video,
                        name="video_radio").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.video_entry = ttk.Entry(self.input_frame, width=50)
        self.video_entry.grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(self.input_frame, text=LANGUAGES[self.language]["browse"], command=self.browse_video,
                   name="video_browse").grid(row=0, column=2, padx=5, pady=2)
        ttk.Radiobutton(self.input_frame, text=LANGUAGES[self.language]["camera"],
                        variable=self.input_var, value="Camera", command=self.toggle_input_camera,
                        name="camera_radio").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.camera_var = tk.StringVar(value=self.camera_devices[0][0] if self.camera_devices else "No devices found")
        self.camera_menu = ttk.OptionMenu(self.input_frame, self.camera_var, self.camera_var.get(),
                                          *[d[0] for d in self.camera_devices])
        self.camera_menu.grid(row=1, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.camera_menu.config(state='disabled')

        # 鱼缸类型选择
        self.tank_type_frame = ttk.LabelFrame(self.main_interface_frame, text=LANGUAGES[self.language]["tank_type"],
                                              padding="5")
        self.tank_type_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        self.tank_shape_var = tk.StringVar(value="rectangle")
        ttk.Radiobutton(self.tank_type_frame, text=LANGUAGES[self.language]["rectangle"],
                        variable=self.tank_shape_var, value="rectangle", name="rect_radio").grid(row=0, column=0,
                                                                                                 padx=10, pady=2, sticky=tk.W)
        ttk.Radiobutton(self.tank_type_frame, text=LANGUAGES[self.language]["trapezoid"],
                        variable=self.tank_shape_var, value="trapezoid", name="trap_radio").grid(row=0, column=1,
                                                                                                 padx=10, pady=2, sticky=tk.W)

        # 选项和输出（采用 2 行×5 列的网格排列）
        self.options_frame = ttk.LabelFrame(self.main_interface_frame, text=LANGUAGES[self.language]["options_output"],
                                            padding="5")
        self.options_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        # 第一列：保存原视频和保存预览视频
        self.save_raw_var = tk.BooleanVar(value=self.config["recording"].get("save_raw_video", False))
        ttk.Checkbutton(self.options_frame, text="Save Raw Video", variable=self.save_raw_var,
                        name="save_raw_video").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.save_preview_var = tk.BooleanVar(value=self.config["features"].get("save_preview_video", False))
        ttk.Checkbutton(self.options_frame, text=LANGUAGES[self.language]["save_preview_video"], variable=self.save_preview_var,
                        name="save_preview_video").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)

        # 其余选项（依次排列到第2～5列，2行共8个位置）
        other_options = [
            ("calculate_speed", LANGUAGES[self.language]["calculate_speed"]),
            ("calculate_top_time", LANGUAGES[self.language]["calculate_top_time"]),
            ("calculate_freeze_time", LANGUAGES[self.language]["calculate_freeze_time"]),
            ("generate_trajectory", LANGUAGES[self.language]["generate_trajectory"]),
            ("generate_heatmap", LANGUAGES[self.language]["generate_heatmap"]),
            ("generate_speed_plot", LANGUAGES[self.language]["generate_speed_plot"]),
            ("generate_timeline", LANGUAGES[self.language]["generate_timeline"]),
            ("calculate_total_displacement", LANGUAGES[self.language]["calculate_total_displacement"]),
        ]
        # 将 8 个选项依次填入，行 = i//4, 列 = (i%4)+1
        self.feature_vars = {}
        for i, (key, label) in enumerate(other_options):
            row = i // 4
            col = (i % 4) + 1
            var = tk.BooleanVar(value=self.config["features"].get(key, False))
            ttk.Checkbutton(self.options_frame, text=label, variable=var, name=f"check_{key}",
                            onvalue=True, offvalue=False).grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            self.feature_vars[key] = var

        # 输出设置
        self.output_frame = ttk.LabelFrame(self.main_interface_frame, text=LANGUAGES[self.language]["options_output"],
                                           padding="5")
        self.output_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["output_path"], name="output_label", width=15,
                  anchor=tk.W).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.output_entry = ttk.Entry(self.output_frame, width=50)
        self.output_entry.insert(0, self.config["paths"]["output_base_path"])
        self.output_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky=(tk.W, tk.E))
        self.output_browse_button = ttk.Button(self.output_frame, text=LANGUAGES[self.language]["browse"],
                                               command=self.browse_output, name="output_browse")
        self.output_browse_button.grid(row=0, column=3, padx=5, pady=2)
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["group_name"], name="group_name_label", width=15,
                  anchor=tk.W).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.group_name_entry = ttk.Entry(self.output_frame, width=20, foreground='grey')
        self.group_name_entry.insert(0, LANGUAGES[self.language]["group_name_placeholder"])
        self.group_name_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.group_name_entry.bind("<FocusIn>", self.clear_placeholder)
        self.group_name_entry.bind("<FocusOut>", self.restore_placeholder)
        self.group_name_entry.bind("<KeyRelease>", self.update_export_preview)
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["sequence_number"], name="sequence_number_label",
                  width=15, anchor=tk.W).grid(row=1, column=2, sticky=tk.W, pady=2)
        self.sequence_spinbox = ttk.Spinbox(self.output_frame, from_=1, to=999, textvariable=self.sequence_number_var,
                                            width=5)
        self.sequence_spinbox.grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        self.sequence_spinbox.bind("<KeyRelease>", self.update_export_preview)
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["export_preview"], name="export_preview_label",
                  width=15, anchor=tk.W).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.export_preview_entry = ttk.Entry(self.output_frame, width=50, state='readonly')
        self.export_preview_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=2, sticky=(tk.W, tk.E))
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["set_record_time"], name="set_record_time_label",
                  width=15, anchor=tk.W).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.set_record_time_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.output_frame, text="", variable=self.set_record_time_var, command=self.toggle_record_time,
                        name="set_record_time_check").grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        self.minutes_var = tk.IntVar(value=self.config["recording"]["max_record_time"] // 60)
        self.seconds_var = tk.IntVar(value=self.config["recording"]["max_record_time"] % 60)
        self.minutes_spinbox = ttk.Spinbox(self.output_frame, from_=0, to=999, textvariable=self.minutes_var, width=5)
        self.minutes_spinbox.grid(row=3, column=2, padx=5, pady=2, sticky=tk.W)
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["minutes"], name="minutes_label", width=10).grid(
            row=3, column=2, padx=(65, 0), pady=2, sticky=tk.W)
        self.seconds_spinbox = ttk.Spinbox(self.output_frame, from_=0, to=59, textvariable=self.seconds_var, width=5)
        self.seconds_spinbox.grid(row=3, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["seconds"], name="seconds_label", width=10).grid(
            row=3, column=3, padx=(65, 0), pady=2, sticky=tk.W)
        ttk.Label(self.output_frame, text=LANGUAGES[self.language]["skip_frames"], name="skip_frames_label", width=15,
                  anchor=tk.W).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.skip_frames_var = tk.IntVar(value=self.config["recording"]["skip_frames"])
        self.skip_frames_spinbox = ttk.Spinbox(self.output_frame, from_=1, to=10, textvariable=self.skip_frames_var, width=5)
        self.skip_frames_spinbox.grid(row=4, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(self.output_frame, text="ⓘ", name="skip_frames_info").grid(row=4, column=2, padx=5, pady=2)
        self.output_frame.children["skip_frames_info"].bind("<Enter>", lambda e: self.show_info("skip_frames"))
        self.output_frame.children["skip_frames_info"].bind("<Leave>", lambda e: self.hide_info())
        self.model_label = ttk.Label(self.output_frame, text=LANGUAGES[self.language]["model"].format(
            os.path.basename(self.config["paths"]["model_path"])), name="model_label")
        self.model_label.grid(row=5, column=0, columnspan=4, pady=5)

        # 控制按钮
        self.control_frame = ttk.Frame(self.main_interface_frame)
        self.control_frame.grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(self.control_frame, text=LANGUAGES[self.language]["start_tracking"], command=self.start_tracking,
                   name="start_btn").grid(row=0, column=0, padx=5)
        ttk.Button(self.control_frame, text=LANGUAGES[self.language]["stop_tracking"], command=self.stop_tracking,
                   name="stop_btn").grid(row=0, column=1, padx=5)

        # 鱼缸设置选项卡
        self.tank_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tank_frame, text=LANGUAGES[self.language]["tank_settings"])
        tank_params = [
            ("scale_factor", self.config["tank"]["scale_factor"]),
            ("trapezoid_width_top", self.config["tank"]["trapezoid"]["real_width_top_mm"]),
            ("trapezoid_width_bottom", self.config["tank"]["trapezoid"]["real_width_bottom_mm"]),
            ("trapezoid_height", self.config["tank"]["trapezoid"]["real_height_mm"]),
            ("rectangle_width", self.config["tank"]["rectangle"]["real_width_mm"]),
            ("rectangle_height", self.config["tank"]["rectangle"]["real_height_mm"])
        ]
        self.tank_entries = {}
        for i, (key, default) in enumerate(tank_params):
            ttk.Label(self.tank_frame, text=LANGUAGES[self.language][key], name=f"tank_label_{key}", width=25,
                      anchor=tk.W).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(self.tank_frame)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Label(self.tank_frame, text="ⓘ", name=f"tank_info_{key}").grid(row=i, column=2, padx=5, pady=2)
            self.tank_entries[key] = entry

        # 图像处理选项卡
        self.img_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.img_frame, text=LANGUAGES[self.language]["image_processing"])
        img_params = [
            ("heatmap_kernel_size",
             f"{self.config['image_processing']['heatmap_kernel_size'][0]},{self.config['image_processing']['heatmap_kernel_size'][1]}"),
            ("heatmap_alpha", self.config["image_processing"]["heatmap_alpha"]),
            ("frame_alpha", self.config["image_processing"]["frame_alpha"]),
            ("font_scale", self.config["image_processing"]["font_scale"]),
            ("font_thickness", self.config["image_processing"]["font_thickness"]),
            ("resize_factor", self.config["image_processing"]["resize_factor"]),
        ]
        self.img_entries = {}
        for i, (key, default) in enumerate(img_params):
            ttk.Label(self.img_frame, text=LANGUAGES[self.language][key], name=f"img_label_{key}", width=25,
                      anchor=tk.W).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(self.img_frame)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Label(self.img_frame, text="ⓘ", name=f"img_info_{key}").grid(row=i, column=2, padx=5, pady=2)
            self.img_entries[key] = entry

        # 冻结检测选项卡
        self.freeze_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.freeze_frame, text=LANGUAGES[self.language]["freeze_detection"])
        freeze_params = [
            ("freeze_speed_threshold", self.config["freeze_detection"]["freeze_speed_threshold"]),
            ("freeze_displacement_threshold", self.config["freeze_detection"]["freeze_displacement_threshold"]),
            ("freeze_wh_change_threshold", self.config["freeze_detection"]["freeze_wh_change_threshold"]),
            ("freeze_confidence_threshold", self.config["freeze_detection"]["freeze_confidence_threshold"]),
            ("freeze_duration_threshold", self.config["freeze_detection"]["freeze_duration_threshold"]),
            ("top_duration_threshold", self.config["freeze_detection"]["top_duration_threshold"]),
            ("window_size", self.config["freeze_detection"]["window_size"]),
        ]
        self.freeze_entries = {}
        for i, (key, default) in enumerate(freeze_params):
            ttk.Label(self.freeze_frame, text=LANGUAGES[self.language][key], name=f"freeze_label_{key}", width=25,
                      anchor=tk.W).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(self.freeze_frame)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2)
            ttk.Label(self.freeze_frame, text="ⓘ", name=f"freeze_info_{key}").grid(row=i, column=2, padx=5, pady=2)
            self.freeze_entries[key] = entry

        # 日志输出选项卡
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text=LANGUAGES[self.language]["log_output"])
        self.log_text = tk.Text(self.log_frame, height=10, width=80)
        self.log_text.grid(row=0, column=0, padx=5, pady=5)
        self.text_handler = self.TextHandler(self.log_text)
        logging.getLogger().addHandler(self.text_handler)

        # 语言选择选项卡
        self.lang_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.lang_frame, text=LANGUAGES[self.language]["language_selection"])
        ttk.Label(self.lang_frame, text="Language:", width=15, anchor=tk.W).grid(row=0, column=0, padx=5, pady=5)
        self.lang_var = tk.StringVar(value="English")
        ttk.OptionMenu(self.lang_frame, self.lang_var, "English", "English", "Chinese",
                       command=self.switch_language).grid(row=0, column=1, padx=5, pady=5)

        self.update_export_preview()

    def show_info(self, key):
        if hasattr(self, 'tooltip') and self.tooltip:
            self.tooltip.destroy()
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{self.root.winfo_pointerx() + 10}+{self.root.winfo_pointery() + 10}")
        ttk.Label(self.tooltip, text=PARAM_INFO[key][self.language], wraplength=200,
                  background="lightyellow", relief="solid", borderwidth=1).pack()

    def hide_info(self):
        if hasattr(self, 'tooltip') and self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def toggle_input_video(self):
        self.video_entry.config(state='normal')
        self.camera_menu.config(state='disabled')
        self.config["paths"]["use_camera"] = False
        self.browse_video()

    def toggle_input_camera(self):
        self.video_entry.config(state='disabled')
        self.camera_menu.config(state='normal')
        self.config["paths"]["use_camera"] = True

    def toggle_record_time(self):
        if self.set_record_time_var.get():
            self.minutes_spinbox.config(state='normal')
            self.seconds_spinbox.config(state='normal')
        else:
            self.minutes_spinbox.config(state='disabled')
            self.seconds_spinbox.config(state='disabled')

    def clear_placeholder(self, event):
        if self.group_name_entry.get() == LANGUAGES[self.language]["group_name_placeholder"]:
            self.group_name_entry.delete(0, tk.END)
            self.group_name_entry.config(foreground='black')

    def restore_placeholder(self, event):
        if not self.group_name_entry.get().strip():
            self.group_name_entry.delete(0, tk.END)
            self.group_name_entry.insert(0, LANGUAGES[self.language]["group_name_placeholder"])
            self.group_name_entry.config(foreground='grey')

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if path:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, path)
            self.config["paths"]["video_path"] = path
            self.update_export_preview()

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)
            self.config["paths"]["output_base_path"] = path
            self.update_export_preview()

    def update_export_preview(self, event=None):
        group_name = self.group_name_entry.get() if self.group_name_entry.get() != LANGUAGES[self.language]["group_name_placeholder"] else "Control"
        sequence_number = self.sequence_number_var.get()
        output_path = self.output_entry.get().rstrip('/')
        preview = f"{output_path}/{group_name}_{sequence_number:03d}.mp4" if output_path and group_name else "N/A"
        self.export_preview_entry.config(state='normal')
        self.export_preview_entry.delete(0, tk.END)
        self.export_preview_entry.insert(0, preview)
        self.export_preview_entry.config(state='readonly')

    def switch_language(self, lang):
        self.language = lang
        self.root.title(LANGUAGES[lang]["title"])
        self.notebook.tab(0, text=LANGUAGES[lang]["main_interface"])
        self.notebook.tab(1, text=LANGUAGES[lang]["tank_settings"])
        self.notebook.tab(2, text=LANGUAGES[lang]["image_processing"])
        self.notebook.tab(3, text=LANGUAGES[lang]["freeze_detection"])
        self.notebook.tab(4, text=LANGUAGES[lang]["log_output"])
        self.notebook.tab(5, text=LANGUAGES[lang]["language_selection"])

        # 更新输入源部分
        self.input_frame.config(text=LANGUAGES[lang]["input_source"])
        self.input_frame.children["video_radio"].config(text=LANGUAGES[lang]["video_file"])
        self.input_frame.children["camera_radio"].config(text=LANGUAGES[lang]["camera"])
        self.input_frame.children["video_browse"].config(text=LANGUAGES[lang]["browse"])

        # 更新鱼缸类型部分
        self.tank_type_frame.config(text=LANGUAGES[lang]["tank_type"])
        self.tank_type_frame.children["rect_radio"].config(text=LANGUAGES[lang]["rectangle"])
        self.tank_type_frame.children["trap_radio"].config(text=LANGUAGES[lang]["trapezoid"])

        # 更新选项和输出区域（遍历所有 Checkbutton）
        self.options_frame.config(text=LANGUAGES[lang]["options_output"])
        for widget in self.options_frame.winfo_children():
            if isinstance(widget, ttk.Checkbutton):
                name = widget.winfo_name()
                # 如果名称中包含“save_raw_video”，则使用语言字典中的条目
                if "save_raw_video" in name:
                    widget.config(text=LANGUAGES[lang].get("save_raw_video", "Save Raw Video"))
                elif "save_preview_video" in name:
                    widget.config(text=LANGUAGES[lang].get("save_preview_video", "Save Preview Video"))
                else:
                    # 其它选项通过 feature_vars 更新（名称中含有关键字即可）
                    for key in self.feature_vars:
                        if key in name:
                            widget.config(text=LANGUAGES[lang].get(key, key))
        # 更新输出设置部分
        self.output_frame.config(text=LANGUAGES[lang]["options_output"])
        self.output_frame.children["output_label"].config(text=LANGUAGES[lang]["output_path"])
        self.output_frame.children["group_name_label"].config(text=LANGUAGES[lang]["group_name"])
        self.output_frame.children["sequence_number_label"].config(text=LANGUAGES[lang]["sequence_number"])
        self.output_frame.children["set_record_time_label"].config(text=LANGUAGES[lang]["set_record_time"])
        self.output_frame.children["minutes_label"].config(text=LANGUAGES[lang]["minutes"])
        self.output_frame.children["seconds_label"].config(text=LANGUAGES[lang]["seconds"])
        self.output_frame.children["skip_frames_label"].config(text=LANGUAGES[lang]["skip_frames"])
        self.output_frame.children["export_preview_label"].config(text=LANGUAGES[lang]["export_preview"])
        self.output_browse_button.config(text=LANGUAGES[lang]["browse"])

        # 更新模型信息和控制按钮
        self.model_label.config(
            text=LANGUAGES[lang]["model"].format(os.path.basename(self.config["paths"]["model_path"])))
        self.control_frame.children["start_btn"].config(text=LANGUAGES[lang]["start_tracking"])
        self.control_frame.children["stop_btn"].config(text=LANGUAGES[lang]["stop_tracking"])

        # 更新鱼缸、图像处理和冻结检测页签中各标签的文本
        for key in self.tank_entries:
            self.tank_frame.children[f"tank_label_{key}"].config(text=LANGUAGES[lang].get(key, key))
        for key in self.img_entries:
            self.img_frame.children[f"img_label_{key}"].config(text=LANGUAGES[lang].get(key, key))
        for key in self.freeze_entries:
            self.freeze_frame.children[f"freeze_label_{key}"].config(text=LANGUAGES[lang].get(key, key))

        self.update_export_preview()

    def update_config(self):
        # 更新保存选项，直接使用专用变量
        self.config["recording"]["save_raw_video"] = self.save_raw_var.get()
        self.config["features"]["save_preview_video"] = self.save_preview_var.get()

        try:
            self.config["tank"]["scale_factor"] = float(self.tank_entries["scale_factor"].get())
            self.config["tank"]["trapezoid"]["real_width_top_mm"] = float(
                self.tank_entries["trapezoid_width_top"].get())
            self.config["tank"]["trapezoid"]["real_width_bottom_mm"] = float(
                self.tank_entries["trapezoid_width_bottom"].get())
            self.config["tank"]["trapezoid"]["real_height_mm"] = float(self.tank_entries["trapezoid_height"].get())
            self.config["tank"]["rectangle"]["real_width_mm"] = float(self.tank_entries["rectangle_width"].get())
            self.config["tank"]["rectangle"]["real_height_mm"] = float(self.tank_entries["rectangle_height"].get())
        except ValueError:
            messagebox.showerror("Error", LANGUAGES[self.language]["error_invalid"].format("Tank Parameter"))
            return False

        try:
            kernel_size = self.img_entries["heatmap_kernel_size"].get().split(',')
            self.config["image_processing"]["heatmap_kernel_size"] = (int(kernel_size[0]), int(kernel_size[1]))
            self.config["image_processing"]["heatmap_alpha"] = float(self.img_entries["heatmap_alpha"].get())
            self.config["image_processing"]["frame_alpha"] = float(self.img_entries["frame_alpha"].get())
            self.config["image_processing"]["font_scale"] = float(self.img_entries["font_scale"].get())
            self.config["image_processing"]["font_thickness"] = int(self.img_entries["font_thickness"].get())
            self.config["image_processing"]["resize_factor"] = float(self.img_entries["resize_factor"].get())
        except ValueError:
            messagebox.showerror("Error",
                                 LANGUAGES[self.language]["error_invalid"].format("Image Processing Parameter"))
            return False

        for key, entry in self.freeze_entries.items():
            try:
                self.config["freeze_detection"][key] = float(entry.get())
            except ValueError:
                messagebox.showerror("Error",
                                     LANGUAGES[self.language]["error_invalid"].format(key.replace("_", " ").title()))
                return False

        # 更新其它功能选项，排除 save_preview_video（已经从专用变量更新了）
        for key, var in self.feature_vars.items():
            if key != "save_preview_video":
                self.config["features"][key] = var.get()

        self.config["paths"]["video_path"] = self.video_entry.get() if not self.config["paths"]["use_camera"] else ""
        self.config["paths"]["output_base_path"] = self.output_entry.get()
        group_name = self.group_name_entry.get() if self.group_name_entry.get() != LANGUAGES[self.language][
            "group_name_placeholder"] else "Control"
        self.config["paths"]["video_output_name"] = group_name
        self.config["tank"]["tank_shape"] = self.tank_shape_var.get()

        if self.config["paths"]["use_camera"]:
            selected_device = self.camera_var.get()
            try:
                index = int(selected_device.split()[-1])
                self.config["paths"]["camera_index"] = index
            except ValueError:
                messagebox.showerror("Error", "Invalid camera index selected")
                return False

        if self.set_record_time_var.get():
            try:
                minutes = self.minutes_var.get()
                seconds = self.seconds_var.get()
                if minutes < 0 or seconds < 0 or seconds >= 60:
                    raise ValueError("Invalid time value")
                self.config["recording"]["max_record_time"] = minutes * 60 + seconds
                self.config["recording"]["skip_frames"] = self.skip_frames_var.get()
            except ValueError:
                messagebox.showerror("Error", LANGUAGES[self.language]["error_invalid"].format("Record Time"))
                return False
        else:
            self.config["recording"]["max_record_time"] = float('inf')
            self.config["recording"]["skip_frames"] = self.skip_frames_var.get()

        return True

    def start_tracking(self):
        if not self.update_config():
            return
        if not self.config["paths"]["use_camera"] and not self.config["paths"]["video_path"]:
            messagebox.showerror("Error", LANGUAGES[self.language]["error_missing"])
            return
        self.running = True
        self.stop_event.clear()  # 清除事件状态
        VideoProcessor.video_sequence = self.sequence_number_var.get()
        self.processor = VideoProcessor(self.config)
        self.tracker = TrackerAnalyzer(self.config, self.processor)
        self.display_process = mp.Process(target=display_process, args=(self.frame_queue, self.stop_event), daemon=True)
        self.display_process.start()
        logging.info("Starting tracking with buffer period...")
        self.running_thread = threading.Thread(target=self._track_with_queue, args=(self.stop_event,), daemon=True)
        self.running_thread.start()

    def _track_with_queue(self, stop_event):
        try:
            self.tracker.track_and_analyze(stop_event, self.frame_queue)
        except Exception as e:
            logging.error(f"Tracking error: {e}")
        finally:
            self.running = False
            self.stop_event.set()
            self.root.after(0, lambda: self.sequence_number_var.set(self.sequence_number_var.get() + 1))
            self.root.after(0, self.update_export_preview)

    def stop_tracking(self):
        if not self.running:
            return
        self.running = False
        self.stop_event.set()

        def cleanup():
            if self.processor:
                self.processor.cleanup()
            if self.running_thread:
                self.running_thread.join(timeout=1.0)
                self.running_thread = None
            if self.display_process:
                self.display_process.join(timeout=1.0)
                if self.display_process.is_alive():
                    self.display_process.terminate()
                self.display_process = None
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            logging.info("Tracking stopped via UI and display process terminated")

        threading.Thread(target=cleanup, daemon=True).start()
        self.root.update_idletasks()

    class TextHandler(logging.Handler):
        def __init__(self, text_widget):
            super().__init__()
            self.text_widget = text_widget

        def emit(self, record):
            msg = self.format(record)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)

def main():
    root = tk.Tk()
    app = TrackerUI(root)
    root.mainloop()

if __name__ == "__main__":
    mp.freeze_support()
    main()
