import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from tracking.config import Config
from tracking.setup_yolo_imcmd import patch_yolo_custom_blocks
from tracking.model import build_sahi_yolo
from tracking.rerank import Reranker
from tracking.detecting_frame import detect_single_frame
from tracking.tracking_video import run_video_tracking

# ==== import các phần phụ thuộc từ repo của bạn ====
cfg = Config()
print(f"[INFO] Device = {cfg.DEVICE}")

# bảo đảm YOLO_DIR trong sys.path
if cfg.YOLO_DIR not in sys.path:
    sys.path.insert(0, cfg.YOLO_DIR)

from tracker import Tracker  # từ repo của bạn
# embeddings utils từ repo của bạn
from embeddings.utils import load_ref_prototype, encode_images, pil_from_bgr, clip_cosine
try:
    from embeddings.utils import encode_patches_batch
except Exception:
    def encode_patches_batch(pil_list, bs=32):
        outs = []
        for i in range(0, len(pil_list), bs):
            outs.append(encode_images(pil_list[i:i+bs]))
        import torch as _torch
        return _torch.cat(outs, dim=0) if outs else _torch.empty(0, 0)

# ==== Patch custom YOLO blocks (IMCMD) ====
patch_yolo_custom_blocks(cfg.IMCMD_DIR)

# ==== Kiểm tra assets / video ====
print(f"[INFO] Video path = {cfg.VIDEO_PATH}")
if not cfg.VIDEO_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy video: {cfg.VIDEO_PATH}")

# ==== Ref prototype ====
ref_proto, ref_feats = load_ref_prototype(cfg.REF_DIR)
print("[INFO] Ref embeddings ready.")

# ==== SAHI + YOLO model ====
model = build_sahi_yolo(cfg)

# ==== Reranker & Tracker ====
reranker = Reranker(cfg, ref_proto, encode_patches_batch, encode_images, pil_from_bgr)
tracker = Tracker()

# ==== Detect single frame trước (tùy thích) ====
try:
    print(f"[INFO] Run single-frame detection on: {cfg.FRAME_PATH}")
    single_res = detect_single_frame(
        cfg, reranker, model, cfg.FRAME_PATH,
        save_vis=False, show_window=cfg.SHOW_VIDEO
    )
    print("[INFO] Single-frame result:", single_res)
except Exception as e:
    print(f"[WARN] Single-frame detection failed: {e}")

# ==== Chạy tracking video ====
# run_video_tracking(cfg, reranker, model, tracker, ref_proto, encode_images, pil_from_bgr)
