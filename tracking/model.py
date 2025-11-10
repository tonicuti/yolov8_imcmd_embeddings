from sahi import AutoDetectionModel
from .config import Config

def build_sahi_yolo(cfg: Config):
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str((cfg.IMCMD_DIR + "\\best.pt")),
        confidence_threshold=cfg.detection_threshold,
        device=cfg.DEVICE,
        image_size=cfg.IMAGE_SIZE
    )
    return model
