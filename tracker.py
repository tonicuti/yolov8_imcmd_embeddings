# tracker.py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small

# DeepSORT (giữ nguyên)
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection


# ---------------------- ReID Encoder (PyTorch) ----------------------
class TorchReIDEncoder(nn.Module):
    """
    Encoder ReID nhẹ (MobileNetV3-Small) -> 128-D L2-normalized.
    Dùng được trực tiếp để thay MARS .pb trong DeepSORT.
    """
    def __init__(self, emb_dim=128, device=None, half=False):
        super().__init__()
        base = mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.backbone = base.features  # (B, C=576, H/32, W/32 với input 128x128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = (half and self.dev.type == "cuda")
        self.to(self.dev)
        if self.half:
            self.half_()

        # Chuẩn hoá giống ImageNet
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        # x: (B,3,128,128)
        f = self.backbone(x)
        f = self.pool(f).flatten(1)    # (B, 576)
        e = self.head(f)               # (B, emb_dim)
        e = F.normalize(e, dim=1)      # L2
        return e

    @torch.no_grad()
    def encode_crops(self, crops_bgr):
        """
        crops_bgr: list of np.ndarray (H,W,3) BGR.
        return: np.ndarray (N, emb_dim)
        """
        if len(crops_bgr) == 0:
            return np.zeros((0, self.head[-1].out_features), dtype=np.float32)

        batch = []
        for c in crops_bgr:
            if c is None or c.size == 0:
                # tạo placeholder đen để tránh lỗi
                c = np.zeros((128, 128, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            t = self.tf(rgb)
            if self.half:
                t = t.half()
            batch.append(t)

        x = torch.stack(batch, 0).to(self.dev)  # (B,3,128,128)
        e = self.forward(x)
        return e.float().cpu().numpy()

    @torch.no_grad()
    def __call__(self, frame_bgr, tlwh_boxes):
        """
        API tương thích với gdet.create_box_encoder:
        - frame_bgr: np.ndarray (H,W,3) BGR
        - tlwh_boxes: np.ndarray (N,4) [x, y, w, h] (float)
        return: features np.ndarray (N, emb_dim), thứ tự tương ứng tlwh_boxes
        """
        H, W = frame_bgr.shape[:2]
        crops = []
        for x, y, w, h in tlwh_boxes:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(W, int(x + w))
            y2 = min(H, int(y + h))
            crop = frame_bgr[y1:y2, x1:x2]
            crops.append(crop)
        feats = self.encode_crops(crops)
        return feats


# ---------------------- Adapter Tracker giữ nguyên interface ----------------------
class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self,
                 max_cosine_distance=0.4,
                 nn_budget=None,
                 device=None,
                 half=False):
        """
        Thay encoder TF MARS bằng TorchReIDEncoder.
        """
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = DeepSortTracker(metric)
        self.encoder = TorchReIDEncoder(emb_dim=128, device=device, half=half)
        self.tracks = []

    def update(self, frame, detections):
        """
        detections: list of [x1, y1, x2, y2, score]
        """
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  # không có detection
            self.update_tracks()
            return

        # xyxy -> tlwh
        bboxes_xyxy = np.asarray([d[:-1] for d in detections], dtype=np.float32)
        scores      = np.asarray([d[-1]  for d in detections], dtype=np.float32)

        tlwh = bboxes_xyxy.copy()
        tlwh[:, 2:] = tlwh[:, 2:] - tlwh[:, 0:2]   # (w,h)
        tlwh[:, 0:2] = bboxes_xyxy[:, 0:2]         # (x,y)

        # ReID features (PyTorch)
        features = self.encoder(frame, tlwh)

        dets = []
        for i in range(len(tlwh)):
            dets.append(Detection(tlwh[i], scores[i], features[i]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox_tlbr = track.to_tlbr()  # [x1,y1,x2,y2]
            tracks.append(Track(track.track_id, bbox_tlbr))
        self.tracks = tracks


class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
