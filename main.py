import os
import random
import time
from pathlib import Path
import sys
import cv2
import torch
import torch.nn.functional as F
import json
from IPython.display import clear_output
import subprocess
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

YOLO_DIR = r"C:\ZaloAI2025\observing\train\yoloV8"   

if YOLO_DIR not in sys.path:
    sys.path.insert(0, YOLO_DIR)

from tracker import Tracker 
import ultralytics.nn.tasks as ytasks
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

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except Exception:
    IN_COLAB = False

SAVE_VIDEO = False 
SHOW_VIDEO = False

# ========== Ultralyitcs setup ==========
IMCMD_DIR = Path(r"C:\ZaloAI2025\observing\train\yoloV8\yolo_imcmd")
sys.path.insert(0, str(IMCMD_DIR.parent))

from yolo_imcmd import modules as immods
ytasks.C2f_CA = immods.C2f_CA 
ytasks.AMFF = immods.AMFF
ytasks.DynamicHeadLite = immods.DynamicHeadLite

# ========== Paths ==========
BASE_DIR = r"C:\ZaloAI2025\observing\train\samples"  
BASE_PATH = rf"{BASE_DIR}\Person1_0"
video_path = rf"{BASE_PATH}\drone_video.mp4"
out_path = rf"{BASE_PATH}\video_tracker_output.mp4"
REF_DIR = rf"{BASE_PATH}\object_images"

# ========== Prepare refs ==========
ref_proto, ref_feats = load_ref_prototype(REF_DIR)   
print("[INFO] Ref embeddings ready.")

print(f"[INFO] Video path = {video_path}")
if not Path(video_path).exists():
    raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

# ========== Params ==========
detection_threshold = 0.35
SLICE_W = 512
SLICE_H = 512
OVERLAP = 0.20
POSTPROC = "NMM"
MATCH_TH = 0.50

SIM_TH   = 0.55
W_DET    = 0.35
W_EMB    = 0.65
PAD_RATIO = 0.04
BATCH_EMB = True    

MIN_AREA = 400   

HARD_NEG_SIM = 0.35

EMA_MOMENTUM = 0.9  

last_good = {
    "bbox": None,          # (x1,y1,x2,y2)
    "center": None,        # (cx, cy)
    "diag": None,          # độ chéo bbox để scale ngưỡng
    "sim": None            # sim của frame trước
}
# Ngưỡng gate
MAX_CENTER_JUMP = 0.6   # cho phép dịch chuyển tối đa 0.6 * đường chéo bbox trước
MIN_SIM_DROP    = 0.20  # không cho sim tụt quá 0.20 so với frame trước (khi đã lock)
LOCK_AFTER_N    = 3     # sau 3 frame đúng thì siết gate hơn
lock_count      = 0
    
neg_bank = []    # list of [D]
NEG_W = 0.25 

# ========== Model (YOLO + SAHI wrapper) ==========
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=rf"{IMCMD_DIR}\best.pt",    
    confidence_threshold=detection_threshold,
    device="cuda:0",
    image_size=640
)

# ========== Tracker ==========
tracker = Tracker()
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255))
          for _ in range(200)]

# ========== Video ==========
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"cv2.VideoCapture không mở được: {video_path}")

# Lấy thông số video gốc
in_fps = cap.get(cv2.CAP_PROP_FPS)
if not in_fps or in_fps <= 0 or in_fps != in_fps:  # NaN check
    in_fps = 30.0  # fallback an toàn
in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (in_w, in_h)

if SHOW_VIDEO and not IN_COLAB:
    win_name = "SAHI + YOLOv8 (visualize)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

writer = None
if SAVE_VIDEO:
    fourcc_try = ['mp4v', 'avc1', 'XVID']
    for cc in fourcc_try:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(out_path), fourcc, in_fps, frame_size)
        if writer.isOpened():
            print(f"[INFO] Ghi video với codec: {cc} -> {out_path}")
            break
    if writer is None or not writer.isOpened():
        cap.release()
        raise RuntimeError("Không khởi tạo được VideoWriter.")
    
# tạo cửa sổ co giãn
if SHOW_VIDEO and not IN_COLAB:
    win_name = "SAHI + YOLOv8 (visualize)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

paused = False
prev_t = time.time()
frame_id = 0

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    interx1, intery1 = max(ax1,bx1), max(ay1,by1)
    interx2, intery2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, interx2-interx1), max(0, intery2-intery1)
    inter = iw*ih
    areaA = (ax2-ax1)*(ay2-ay1); areaB = (bx2-bx1)*(by2-by1)
    return inter/max(1e-6, areaA+areaB-inter)

def add_negative(feat):    # feat đã L2-norm shape [1,D]
    if len(neg_bank) < 64:
        neg_bank.append(feat.squeeze(0).cpu())
    else:
        neg_bank.pop(0); neg_bank.append(feat.squeeze(0).cpu())

def negative_penalty(emb): # emb [D]
    if not neg_bank:
        return 0.0
    negs = torch.stack(neg_bank, dim=0).to(emb.device)  # [K,D]
    s = torch.matmul(negs, emb.reshape(-1,1)).max().item()  # max cosine với negatives
    # scale về [0,1]
    s01 = (max(-1.0, min(1.0, s)) + 1.0)*0.5
    return NEG_W * s01

@torch.inference_mode()
def rerank_and_filter(frame_bgr, dets_xyxy_score, sim_th=SIM_TH):
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
    sim  = clip_cosine(embs, ref_proto)   # [-1,1]
    sim01 = (sim.clamp(-1,1)+1.0)*0.5     # [0,1]

    out = []
    for i, (x1,y1,x2,y2,sc) in enumerate(boxes):
        s = float(sim01[i].item())
        if s < HARD_NEG_SIM and sc > detection_threshold + 0.1:
            if last_good["bbox"] is None or iou((x1,y1,x2,y2), last_good["bbox"]) < 0.2:
                add_negative(embs[i:i+1])
        
        pen = negative_penalty(embs[i])
        sc_r = W_DET*float(sc) + W_EMB*s - pen

        # --- temporal gating nếu đã có last_good ---
        accept = True
        if last_good["bbox"] is not None:
            cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
            gcx, gcy = last_good["center"]
            dist = ((cx-gcx)**2 + (cy-gcy)**2)**0.5
            max_jump = MAX_CENTER_JUMP * last_good["diag"]
            
            if lock_count >= LOCK_AFTER_N:
                tight = 0.85
            else:
                tight = 1.0
                
            if dist > tight*max_jump:
                accept = False
            if last_good["sim"] is not None and (last_good["sim"] - s) > MIN_SIM_DROP:
                accept = False

            # cấm nhảy hộp quá lạ: iou với bbox trước ~0
            if iou((x1,y1,x2,y2), last_good["bbox"]) < 0.05 and dist > 1.2*max_jump:
                accept = False

        if s >= sim_th and accept:
            out.append([x1,y1,x2,y2,sc_r,s])

    out.sort(key=lambda x: x[4], reverse=True)
    return out

def update_last_good(chosen):
    # chosen: [x1,y1,x2,y2,score_r,sim]
    if not chosen:
        return
    x1,y1,x2,y2,_,s = chosen
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    diag = ((x2-x1)**2 + (y2-y1)**2)**0.5
    last_good["bbox"] = (x1,y1,x2,y2)
    last_good["center"] = (cx, cy)
    last_good["diag"] = max(1.0, diag)
    last_good["sim"] = s

@torch.inference_mode()
def maybe_update_proto(frame_bgr, box, sim_now):
    if sim_now < max(0.65, SIM_TH+0.05):
        return
    x1,y1,x2,y2 = map(int, box)
    crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0: 
        return
    feat = encode_images([pil_from_bgr(crop)])[0:1]  # [1,D]
    feat = F.normalize(feat, dim=1)
    global ref_proto
    ref_proto = F.normalize(EMA_MOMENTUM*ref_proto + (1-EMA_MOMENTUM)*feat, dim=1)

video_id = Path(video_path).stem

# Nơi lưu file submission
submission_path = Path(BASE_PATH) / "submission.json"

# Khung dữ liệu submission cho video hiện tại
submission_rec = {
    "video_id": video_id,
    "detections": [
        {"bboxes": []}  # sẽ append từng bbox theo frame
    ]
}

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[INFO] Hết video hoặc đọc frame lỗi.")
            break
        frame_id += 1

        # SAHI sliced prediction
        pred = get_sliced_prediction(
            image=frame,
            detection_model=model,
            slice_height=SLICE_H,
            slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            perform_standard_pred=False,
            postprocess_type=POSTPROC,
            postprocess_match_threshold=MATCH_TH,
            verbose=0
        )

        # SAHI -> detections cho Tracker
        detections = []
        for obj in pred.object_prediction_list:
            x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
            x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
            if (x2-x1)*(y2-y1) < MIN_AREA:
                continue
    
            score = float(obj.score.value)
            if score >= detection_threshold:
                detections.append([x1, y1, x2, y2, score])
                
        SIM_TH_bak, MAX_CENTER_JUMP_bak = SIM_TH, MAX_CENTER_JUMP

        if lock_count <= 1:               
            sim_th_use = max(0.45, SIM_TH - 0.08)
            max_center_jump_use = MAX_CENTER_JUMP * 1.4
        else:
            sim_th_use = SIM_TH
            max_center_jump_use = MAX_CENTER_JUMP

        dets = rerank_and_filter(frame, detections, sim_th=sim_th_use)
        SIM_TH, MAX_CENTER_JUMP = SIM_TH_bak, MAX_CENTER_JUMP_bak

        best = dets[0] if len(dets) > 0 else None
        if best:
            update_last_good(best)
            tracker_dets = [[best[0], best[1], best[2], best[3], best[4]]]
            if lock_count >= LOCK_AFTER_N and (best[2]-best[0])*(best[3]-best[1]) >= MIN_AREA*2:
                maybe_update_proto(frame, best[:4], best[5])
            lock_count = min(lock_count + 1, 10)

            x1, y1, x2, y2 = map(int, best[:4])
            submission_rec["detections"][0]["bboxes"].append({
                "frame": int(frame_id),  # frame_id hiện đang đếm từ 1
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
        else:
            tracker_dets = []  # không gán nhầm
            lock_count = max(lock_count - 1, 0)
            
            if lock_count == 0:
                last_good["bbox"] = None
                last_good["center"] = None
                last_good["diag"] = None
                last_good["sim"] = None
                neg_bank.clear()

        # update tracker
        tracker.update(frame, tracker_dets)

        # vẽ bbox + id
        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            tid = track.track_id
            color = colors[tid % len(colors)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            sim_txt = ""
            for d in dets:
                if abs(d[0]-x1)+abs(d[1]-y1)+abs(d[2]-x2)+abs(d[3]-y2) < 8:  # gần trùng bbox
                    sim_txt = f" sim:{d[5]:.2f}"
                    break
            cv2.putText(frame, f'ID {tid}{sim_txt}', (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # FPS display
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t))
        prev_t = now
        cv2.putText(frame, f"Frame: {frame_id} | FPS: {fps:.1f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"lock:{lock_count} simTH:{SIM_TH:.2f}",
            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,0), 2, cv2.LINE_AA)

        
        if SHOW_VIDEO:
            if IN_COLAB:
                clear_output(wait=True)
                cv2_imshow(frame) 
                time.sleep(1 / 30.0)
            else:
                cv2.imshow(win_name, frame)

        if SAVE_VIDEO and writer is not None:
            writer.write(frame)

        
    # phím tắt
    if not IN_COLAB:
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            print("[INFO] Thoát.")
            break
        elif key == ord('p'):
            paused = not paused
    else:
        pass

with open(submission_path, "w", encoding="utf-8") as f:
    json.dump([submission_rec], f, ensure_ascii=False, indent=2)

print(f"[OK] Submission saved to: {submission_path}")
print(f"[INFO] Total bboxes: {len(submission_rec['detections'][0]['bboxes'])}")

cap.release()
if writer is not None:
    writer.release()
if SHOW_VIDEO and not IN_COLAB:
    cv2.destroyAllWindows()
