from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # ===== Paths =====
    YOLO_DIR: str = r"C:\ZaloAI2025\observing\train\yoloV8_"
    IMCMD_DIR: str = r"C:\ZaloAI2025\observing\train\yolov8_\yolo_imcmd"
    BASE_DIR: str = r"C:\ZaloAI2025\observing\train\samples"
    BASE_PATH_NAME: str = "Lifering_1"   
    VIDEO_NAME: str = "drone_video.mp4"
    FRAME_PATH: str = r"C:\ZaloAI2025\observing\train\extract_frames\frames\Lifering_1_drone_video\drone_video_0101.jpg"

    # ===== Runtime =====
    SHOW_VIDEO: bool = True
    SAVE_VIDEO: bool = False
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== SAHI / YOLO =====
    detection_threshold: float = 0.35
    SLICE_W: int = 512
    SLICE_H: int = 512
    OVERLAP: float = 0.20
    POSTPROC: str = "NMM"
    MATCH_TH: float = 0.50
    IMAGE_SIZE: int = 640

    # ===== CLIP Re-rank =====
    SIM_TH: float = 0.55
    W_DET: float = 0.35
    W_EMB: float = 0.65
    PAD_RATIO: float = 0.04
    BATCH_EMB: bool = True
    MIN_AREA: int = 400
    HARD_NEG_SIM: float = 0.35
    EMA_MOMENTUM: float = 0.9

    # ===== Temporal gates =====
    MAX_CENTER_JUMP: float = 0.6
    MIN_SIM_DROP: float = 0.20
    LOCK_AFTER_N: int = 3

    # ===== Negatives bank =====
    NEG_W: float = 0.25

    # ===== Derived =====
    @property
    def BASE_PATH(self) -> Path:
        return Path(self.BASE_DIR) / self.BASE_PATH_NAME

    @property
    def VIDEO_PATH(self) -> Path:
        return self.BASE_PATH / self.VIDEO_NAME

    @property
    def OUT_VIDEO_PATH(self) -> Path:
        return self.BASE_PATH / "video_tracker_output.mp4"

    @property
    def REF_DIR(self) -> Path:
        return self.BASE_PATH / "object_images"
