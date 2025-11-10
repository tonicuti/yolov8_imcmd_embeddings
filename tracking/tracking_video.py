import time
import json
import cv2
import torch
from pathlib import Path
from .config import Config
from .rerank import Reranker
from .gates import update_last_good

from sahi.predict import get_sliced_prediction

def run_video_tracking(cfg: Config, reranker: Reranker, model, tracker, ref_proto, encode_images, pil_from_bgr):
    cap = cv2.VideoCapture(str(cfg.VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture không mở được: {cfg.VIDEO_PATH}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 0 or in_fps != in_fps:
        in_fps = 30.0
    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (in_w, in_h)

    writer = None
    if cfg.SAVE_VIDEO:
        for cc in ['mp4v', 'avc1', 'XVID']:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(str(cfg.OUT_VIDEO_PATH), fourcc, in_fps, frame_size)
            if writer.isOpened():
                print(f"[INFO] Ghi video với codec: {cc} -> {cfg.OUT_VIDEO_PATH}")
                break
        if writer is None or not writer.isOpened():
            cap.release()
            raise RuntimeError("Không khởi tạo được VideoWriter.")

    if cfg.SHOW_VIDEO:
        win_name = "SAHI + YOLOv8 (visualize)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

    submission_path = cfg.BASE_PATH / "submission.json"
    submission_rec = {"video_id": cfg.VIDEO_PATH.stem, "detections": [{"bboxes": []}]}

    prev_t = time.time()
    frame_id = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[INFO] Hết video hoặc đọc frame lỗi.")
                break
            frame_id += 1

            pred = get_sliced_prediction(
                image=frame,
                detection_model=model,
                slice_height=cfg.SLICE_H,
                slice_width=cfg.SLICE_W,
                overlap_height_ratio=cfg.OVERLAP,
                overlap_width_ratio=cfg.OVERLAP,
                perform_standard_pred=False,
                postprocess_type=cfg.POSTPROC,
                postprocess_match_threshold=cfg.MATCH_TH,
                verbose=0
            )

            detections = []
            for obj in pred.object_prediction_list:
                x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
                x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
                if (x2-x1)*(y2-y1) < cfg.MIN_AREA:
                    continue
                score = float(obj.score.value)
                if score >= cfg.detection_threshold:
                    detections.append([x1, y1, x2, y2, score])

            # adaptive sim threshold khi chưa lock
            if reranker.state.lock_count <= 1:
                sim_th_use = max(0.45, cfg.SIM_TH - 0.08)
            else:
                sim_th_use = cfg.SIM_TH

            dets = reranker.rerank_and_filter(frame, detections, sim_th=sim_th_use)
            best = dets[0] if dets else None

            if best:
                update_last_good(reranker.state, best)
                tracker_dets = [[best[0], best[1], best[2], best[3], best[4]]]
                if (reranker.state.lock_count >= cfg.LOCK_AFTER_N and
                    (best[2]-best[0])*(best[3]-best[1]) >= cfg.MIN_AREA*2):
                    # có thể cập nhật EMA proto nếu cần: dùng trực tiếp trong main nếu muốn
                    pass
                reranker.state.lock_count = min(reranker.state.lock_count + 1, 10)

                x1, y1, x2, y2 = map(int, best[:4])
                submission_rec["detections"][0]["bboxes"].append({
                    "frame": int(frame_id),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
            else:
                tracker_dets = []
                reranker.state.lock_count = max(reranker.state.lock_count - 1, 0)
                if reranker.state.lock_count == 0:
                    reranker.reset()

            tracker.update(frame, tracker_dets)

            # draw
            for track in tracker.tracks:
                x1, y1, x2, y2 = track.bbox
                tid = track.track_id
                color = (0,255,0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                sim_txt = ""
                for d in dets:
                    if abs(d[0]-x1)+abs(d[1]-y1)+abs(d[2]-x2)+abs(d[3]-y2) < 8:
                        sim_txt = f" sim:{d[5]:.2f}"
                        break
                cv2.putText(frame, f'ID {tid}{sim_txt}', (int(x1), int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            cv2.putText(frame, f"Frame: {frame_id} | FPS: {fps:.1f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)
            cv2.putText(frame, f"lock:{reranker.state.lock_count} simTH:{cfg.SIM_TH:.2f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,220,0), 2)

            if cfg.SHOW_VIDEO:
                cv2.imshow("SAHI + YOLOv8 (visualize)", frame)
            if cfg.SAVE_VIDEO and writer is not None:
                writer.write(frame)

        # keys
        key = cv2.waitKey(1) & 0xFF if cfg.SHOW_VIDEO else 255
        if key in (27, ord('q')):
            print("[INFO] Thoát.")
            break
        elif key == ord('p'):
            paused = not paused

    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump([submission_rec], f, ensure_ascii=False, indent=2)
    print(f"[OK] Submission saved to: {submission_path}")
    print(f"[INFO] Total bboxes: {len(submission_rec['detections'][0]['bboxes'])}")

    cap.release()
    if cfg.SHOW_VIDEO:
        cv2.destroyAllWindows()
    if cfg.SAVE_VIDEO and writer is not None:
        writer.release()
