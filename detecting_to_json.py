# detect_frames_to_json.py
import os, re, json, time, sys
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# ====== CONFIG (chỉnh theo máy bạn) ======
FRAMES_ROOT  = Path(r"C:\ZaloAI2025\observing\train\extract_frames\frames_public_test")
SAMPLES_ROOT = Path(r"C:\ZaloAI2025\public_test\public_test\samples")

# IMCMD (YOLO custom modules + checkpoint best.pt)
IMCMD_DIR = Path(r"C:\ZaloAI2025\observing\train\yolov8_\yolo_imcmd")
YOLO_DIR  = Path(r"C:\ZaloAI2025\observing\train\yoloV8_")  # để import tracker/embeddings nếu cần

# Output submission
SUBMISSION_PATH = Path(r"C:\ZaloAI2025\observing\train\yolov8_\submission.json")

# Runtime
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_PROGRESS = True

# SAHI / YOLO params
DETECTION_TH = 0.35
SLICE_W = 512
SLICE_H = 512
OVERLAP = 0.20
POSTPROC = "NMM"
MATCH_TH = 0.50
IMAGE_SIZE = 640

# CLIP Re-rank params
SIM_TH   = 0.55
W_DET    = 0.35
W_EMB    = 0.65
PAD_RATIO = 0.04
BATCH_EMB = True
MIN_AREA = 400
HARD_NEG_SIM = 0.35
EMA_MOMENTUM = 0.9

# Temporal gates
MAX_CENTER_JUMP = 0.6
MIN_SIM_DROP    = 0.20
LOCK_AFTER_N    = 3

# Negatives
NEG_W = 0.25
NEG_BANK_MAXLEN = 64
# ==========================================

# --- sys.path để dùng module repo của bạn ---
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))
if str(IMCMD_DIR.parent) not in sys.path:
    sys.path.insert(0, str(IMCMD_DIR.parent))

# patch YOLO custom blocks
import ultralytics.nn.tasks as ytasks
from yolo_imcmd import modules as immods
ytasks.C2f_CA = immods.C2f_CA
ytasks.AMFF = immods.AMFF
ytasks.DynamicHeadLite = immods.DynamicHeadLite

# SAHI
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# embeddings utils bạn đã có sẵn
from embeddings.utils import load_ref_prototype, encode_images, pil_from_bgr
try:
    from embeddings.utils import encode_patches_batch
except Exception:
    def encode_patches_batch(pil_list, bs=32):
        outs = []
        for i in range(0, len(pil_list), bs):
            outs.append(encode_images(pil_list[i:i+bs]))
        import torch as _torch
        return _torch.cat(outs, dim=0) if outs else _torch.empty(0, 0)

print(f"[INFO] Device = {DEVICE}")

# ===== Build SAHI+YOLO model (Ultralytics) =====
det_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=str(IMCMD_DIR / "best.pt"),
    confidence_threshold=DETECTION_TH,
    device=DEVICE,
    image_size=IMAGE_SIZE
)

# ===== Helpers =====
def natural_key(p: Path) -> Tuple:
    """Sắp xếp frame theo số: 'video_0001.jpg' < 'video_0010.jpg'"""
    s = p.name
    return tuple(int(t) if t.isdigit() else t.lower()
                 for t in re.split(r'(\d+)', s))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    interx1, intery1 = max(ax1,bx1), max(ay1,by1)
    interx2, intery2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, interx2-interx1), max(0, intery2-intery1)
    inter = iw*ih
    areaA = (ax2-ax1)*(ay2-ay1); areaB = (bx2-bx1)*(by2-by1)
    return inter/max(1e-6, areaA+areaB-inter)

class NegBank:
    def __init__(self, maxlen=64):
        self.maxlen = maxlen
        self.bank: List[torch.Tensor] = []
    def add(self, feat_1xD: torch.Tensor):
        v = feat_1xD.squeeze(0).detach().cpu()
        if len(self.bank) >= self.maxlen:
            self.bank.pop(0)
        self.bank.append(v)
    def penalty(self, emb_D: torch.Tensor) -> float:
        if not self.bank:
            return 0.0
        negs = torch.stack(self.bank, dim=0).to(emb_D.device)
        s = torch.matmul(negs, emb_D.reshape(-1,1)).max().item()  # cosine max
        s01 = (max(-1.0, min(1.0, s)) + 1.0)*0.5
        return NEG_W * s01

class TemporalState:
    def __init__(self):
        self.bbox: Optional[Tuple[float,float,float,float]] = None
        self.center: Optional[Tuple[float,float]] = None
        self.diag: Optional[float] = None
        self.sim: Optional[float] = None
        self.lock_count: int = 0
    def reset(self):
        self.__init__()

def update_last_good(state: TemporalState, chosen):
    x1,y1,x2,y2,_,s = chosen
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    diag = ((x2-x1)**2 + (y2-y1)**2)**0.5
    state.bbox = (x1,y1,x2,y2)
    state.center = (cx, cy)
    state.diag = max(1.0, diag)
    state.sim = s

@torch.inference_mode()
def rerank_and_filter(frame_bgr, dets_xyxy_score, ref_proto, neg_bank: NegBank, state: TemporalState, sim_th=SIM_TH):
    if len(dets_xyxy_score) == 0:
        return []

    H, W = frame_bgr.shape[:2]
    crops, boxes = [], []
    for (x1, y1, x2, y2, sc) in dets_xyxy_score:
        w, h = x2-x1, y2-y1
        px, py = int(PAD_RATIO*w), int(PAD_RATIO*h)
        xx1, yy1 = max(0, x1-px), max(0, y1-py)
        xx2, yy2 = min(W, x2+px), min(H, y2+py)
        if xx2 <= xx1 or yy2 <= yy1: 
            continue
        crop = frame_bgr[yy1:yy2, xx1:xx2]
        crops.append(pil_from_bgr(crop))
        boxes.append([x1, y1, x2, y2, sc])

    if not crops:
        return []

    embs = encode_patches_batch(crops, bs=32) if BATCH_EMB else encode_images(crops)
    # cosine với prototype
    e = F.normalize(embs, dim=1)
    p = F.normalize(ref_proto, dim=1)
    sim = (e @ p.T).squeeze(-1).clamp(-1, 1)          # [-1,1]
    sim01 = (sim + 1.0)*0.5                            # [0,1]

    out = []
    for i, (x1,y1,x2,y2,sc) in enumerate(boxes):
        s = float(sim01[i].item())
        if s < HARD_NEG_SIM and sc > DETECTION_TH + 0.1:
            if state.bbox is None or iou((x1,y1,x2,y2), state.bbox) < 0.2:
                neg_bank.add(embs[i:i+1])

        pen = neg_bank.penalty(embs[i])
        sc_r = W_DET*float(sc) + W_EMB*s - pen

        accept = True
        if state.bbox is not None:
            cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
            gcx, gcy = state.center
            dist = ((cx-gcx)**2 + (cy-gcy)**2)**0.5
            max_jump = MAX_CENTER_JUMP * state.diag
            tight = 0.85 if state.lock_count >= LOCK_AFTER_N else 1.0
            if dist > tight*max_jump:
                accept = False
            if state.sim is not None and (state.sim - s) > MIN_SIM_DROP:
                accept = False
            if iou((x1,y1,x2,y2), state.bbox) < 0.05 and dist > 1.2*max_jump:
                accept = False

        if s >= sim_th and accept:
            out.append([x1,y1,x2,y2,sc_r,s])

    out.sort(key=lambda x: x[4], reverse=True)
    return out

def infer_ref_dir_for_video(video_id: str) -> Path:
    """
    Xác định thư mục ref cho từng loại video.
    - Nếu video_id dạng 'Backpack_0_drone_video' hoặc '... .mp4' → lấy prefix trước '_drone_video'
      => REF_DIR = samples/<prefix>/object_images
    - Nếu không chứa '_drone_video', fallback: samples/<video_id>/object_images
    """
    base_name = video_id.replace(".mp4", "")
    prefix = base_name.split("_drone_video")[0] if "_drone_video" in base_name else base_name
    ref_dir = SAMPLES_ROOT / prefix / "object_images"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ref cho video {video_id}: {ref_dir}")
    return ref_dir

def parse_frame_index(fname: str) -> Optional[int]:
    """Kỳ vọng tên file có dạng ..._<frame>.jpg, ví dụ 'drone_video_3426.jpg'"""
    m = re.search(r'(\d+)(?=\.\w+$)', fname)
    return int(m.group(1)) if m else None

def process_video_frames_folder(video_dir: Path) -> Dict:
    """
    Duyệt 1 folder frames của 1 video → trả về record submission theo schema.
    video_id = tên folder (bạn có thể sửa theo quy ước riêng).
    """
    video_id = video_dir.name

    # 1) nạp ref tương ứng của video
    ref_dir = infer_ref_dir_for_video(video_id)
    ref_proto, _ = load_ref_prototype(ref_dir)
    print(f"[INFO] Video={video_id} | REF_DIR={ref_dir}")

    # 2) liệt kê frames
    frames = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")],
        key=natural_key
    )
    if not frames:
        print(f"[WARN] No frames in {video_dir}")
        return {"video_id": video_id, "detections": []}

    neg_bank = NegBank(NEG_BANK_MAXLEN)
    state = TemporalState()

    bboxes_out: List[Dict] = []
    t0 = time.time()

    for idx, fp in enumerate(frames, 1):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        pred = get_sliced_prediction(
            image=img,
            detection_model=det_model,
            slice_height=SLICE_H,
            slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            perform_standard_pred=False,
            postprocess_type=POSTPROC,
            postprocess_match_threshold=MATCH_TH,
            verbose=0
        )
        detections = []
        for obj in pred.object_prediction_list:
            x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
            x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
            if (x2-x1)*(y2-y1) < MIN_AREA:
                continue
            sc = float(obj.score.value)
            if sc >= DETECTION_TH:
                detections.append([x1, y1, x2, y2, sc])

        # adaptive sim threshold khi chưa lock
        sim_th_use = max(0.45, SIM_TH - 0.08) if state.lock_count <= 1 else SIM_TH
        dets = rerank_and_filter(img, detections, ref_proto, neg_bank, state, sim_th=sim_th_use)
        best = dets[0] if dets else None

        # frame index từ tên file hoặc fallback = thứ tự
        fidx = parse_frame_index(fp.name) or idx

        if best:
            update_last_good(state, best)
            state.lock_count = min(state.lock_count + 1, 10)
            x1,y1,x2,y2 = map(int, best[:4])
            bboxes_out.append({"frame": int(fidx), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        else:
            state.lock_count = max(state.lock_count - 1, 0)
            if state.lock_count == 0:
                state.reset()

        if SHOW_PROGRESS and idx % 200 == 0:
            dt = time.time() - t0
            print(f"[..] {video_id}: processed {idx}/{len(frames)} frames in {dt:.1f}s")

    dets_field = [{"bboxes": bboxes_out}] if bboxes_out else []
    return {"video_id": video_id, "detections": dets_field}

def main():
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"FRAMES_ROOT not found: {FRAMES_ROOT}")
    if not IMCMD_DIR.exists():
        raise FileNotFoundError(f"IMCMD_DIR not found: {IMCMD_DIR}")
    if not (IMCMD_DIR / "best.pt").exists():
        raise FileNotFoundError(f"best.pt not found: {IMCMD_DIR/'best.pt'}")

    video_dirs = [d for d in FRAMES_ROOT.iterdir() if d.is_dir()]
    if not video_dirs:
        print(f"[WARN] No video frame folders under {FRAMES_ROOT}")
        return

    results = []
    for d in sorted(video_dirs, key=lambda p: p.name.lower()):
        print(f"[INFO] Processing frames folder: {d}")
        try:
            rec = process_video_frames_folder(d)
            results.append(rec)
        except Exception as e:
            print(f"[ERR] {d.name}: {e}")
            # vẫn push record trống để đảm bảo "Every provided video must appear"
            results.append({"video_id": d.name, "detections": []})

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBMISSION_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote submission to: {SUBMISSION_PATH}")
    print(f"[OK] Total videos: {len(results)}")

if __name__ == "__main__":
    main()
