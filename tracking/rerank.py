from typing import List
import torch
import torch.nn.functional as F
import numpy as np
from .config import Config
from .gates import iou, NegBank, TemporalState

class Reranker:
    def __init__(self, cfg: Config, ref_proto: torch.Tensor,
                 encode_patches_batch, encode_images, pil_from_bgr):
        self.cfg = cfg
        self.ref_proto = ref_proto
        self.encode_patches_batch = encode_patches_batch
        self.encode_images = encode_images
        self.pil_from_bgr = pil_from_bgr
        self.neg_bank = NegBank(maxlen=64)
        self.state = TemporalState()

    def reset(self):
        self.state = TemporalState()
        self.neg_bank = NegBank(maxlen=64)

    @torch.inference_mode()
    def rerank_and_filter(self, frame_bgr, dets_xyxy_score, sim_th=None):
        if len(dets_xyxy_score) == 0:
            return []

        H, W = frame_bgr.shape[:2]
        crops, boxes = [], []
        PAD = self.cfg.PAD_RATIO

        for (x1, y1, x2, y2, sc) in dets_xyxy_score:
            w, h = x2-x1, y2-y1
            px, py = int(PAD*w), int(PAD*h)
            xx1, yy1 = max(0, x1-px), max(0, y1-py)
            xx2, yy2 = min(W, x2+px), min(H, y2+py)
            if xx2 <= xx1 or yy2 <= yy1:
                continue
            crop = frame_bgr[yy1:yy2, xx1:xx2]
            crops.append(self.pil_from_bgr(crop))
            boxes.append([x1, y1, x2, y2, sc])

        if not crops:
            return []

        embs = (self.encode_patches_batch(crops, bs=32)
                if self.cfg.BATCH_EMB else self.encode_images(crops))
        sim  = torch.clamp(self._clip_cosine(embs, self.ref_proto), -1, 1)  # [-1,1]
        sim01 = (sim + 1.0) * 0.5

        out = []
        for i, (x1,y1,x2,y2,sc) in enumerate(boxes):
            s = float(sim01[i].item())
            # Hard-negative mining
            if s < self.cfg.HARD_NEG_SIM and sc > self.cfg.detection_threshold + 0.1:
                if self.state.bbox is None or iou((x1,y1,x2,y2), self.state.bbox) < 0.2:
                    self.neg_bank.add(embs[i:i+1])

            pen = self.neg_bank.penalty(embs[i], self.cfg.NEG_W)
            sc_r = self.cfg.W_DET*float(sc) + self.cfg.W_EMB*s - pen

            # temporal gating
            accept = True
            if self.state.bbox is not None:
                cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
                gcx, gcy = self.state.center
                dist = ((cx-gcx)**2 + (cy-gcy)**2)**0.5
                max_jump = self.cfg.MAX_CENTER_JUMP * self.state.diag

                tight = 0.85 if self.state.lock_count >= self.cfg.LOCK_AFTER_N else 1.0
                if dist > tight*max_jump:
                    accept = False
                if self.state.sim is not None and (self.state.sim - s) > self.cfg.MIN_SIM_DROP:
                    accept = False
                if iou((x1,y1,x2,y2), self.state.bbox) < 0.05 and dist > 1.2*max_jump:
                    accept = False

            if s >= (self.cfg.SIM_TH if sim_th is None else sim_th) and accept:
                out.append([x1,y1,x2,y2,sc_r,s])

        out.sort(key=lambda x: x[4], reverse=True)
        return out

    @staticmethod
    def _clip_cosine(embs: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        # embs: [N,D], proto: [1,D]; cả hai đã/hoặc chưa norm => chuẩn hoá
        e = F.normalize(embs, dim=1)
        p = F.normalize(proto, dim=1)
        return (e @ p.T).squeeze(-1)
