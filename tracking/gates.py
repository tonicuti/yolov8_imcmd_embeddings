from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F

class TemporalState:
    def __init__(self):
        self.bbox: Optional[Tuple[float, float, float, float]] = None
        self.center: Optional[Tuple[float, float]] = None
        self.diag: Optional[float] = None
        self.sim: Optional[float] = None
        self.lock_count: int = 0

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

    def penalty(self, emb_D: torch.Tensor, NEG_W: float) -> float:
        if not self.bank:
            return 0.0
        negs = torch.stack(self.bank, dim=0).to(emb_D.device)  # [K,D]
        s = torch.matmul(negs, emb_D.reshape(-1,1)).max().item()  # cosine max
        s01 = (max(-1.0, min(1.0, s)) + 1.0)*0.5
        return NEG_W * s01

def update_last_good(state: TemporalState, chosen):
    # chosen: [x1,y1,x2,y2,score_r,sim]
    x1,y1,x2,y2,_,s = chosen
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    diag = ((x2-x1)**2 + (y2-y1)**2)**0.5
    state.bbox = (x1,y1,x2,y2)
    state.center = (cx, cy)
    state.diag = max(1.0, diag)
    state.sim = s

def ema_update_proto(ref_proto: torch.Tensor, frame_crop_bgr, box, encode_images, pil_from_bgr, momentum=0.9):
    import numpy as np
    import torch
    x1,y1,x2,y2 = map(int, box)
    crop = frame_crop_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0:
        return ref_proto
    feat = encode_images([pil_from_bgr(crop)])[0:1]  # [1,D]
    feat = F.normalize(feat, dim=1)
    ref_proto = F.normalize(momentum*ref_proto + (1-momentum)*feat, dim=1)
    return ref_proto
