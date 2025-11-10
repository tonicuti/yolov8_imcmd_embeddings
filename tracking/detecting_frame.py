import cv2
import torch
from pathlib import Path
from sahi.predict import get_sliced_prediction
from .config import Config
from .rerank import Reranker

@torch.inference_mode()
def detect_single_frame(
    cfg: Config,
    reranker: Reranker,
    model,
    img_path: str,
    save_vis=True,
    show_window=None
):
    """
    Chạy SAHI + YOLO + CLIP rerank trên 1 ảnh tĩnh.
    KHÔNG ghi JSON (chỉ trả kết quả trong Python và optionally ảnh trực quan).
    """
    if show_window is None:
        show_window = cfg.SHOW_VIDEO

    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {p}")

    reranker.reset()

    frame = cv2.imread(str(p))
    if frame is None:
        raise RuntimeError(f"cv2.imread đọc lỗi ảnh: {p}")

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

    dets = reranker.rerank_and_filter(frame, detections)

    vis = frame.copy()
    result = {"bbox": None, "score": None, "sim": None, "out_img": None}
    if dets:
        x1,y1,x2,y2,score_r,sim = dets[0]
        result["bbox"] = (int(x1), int(y1), int(x2), int(y2))
        result["score"] = float(score_r)
        result["sim"] = float(sim)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(vis, f"scoreR:{score_r:.2f} sim:{sim:.2f}",
                    (int(x1), max(20, int(y1)-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        cv2.putText(vis, "NO DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if save_vis:
        out_img = str(p.with_name(p.stem + "_det.jpg"))
        cv2.imwrite(out_img, vis)
        result["out_img"] = out_img
        print(f"[OK] Saved visualization: {out_img}")

    if show_window:
        win = "Single-frame detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, vis)
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    return result
