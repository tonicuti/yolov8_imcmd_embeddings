import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image

EMB_MODEL_NAME = "ViT-B-32"
EMB_PRETRAINED = "laion2b_s34b_b79k"


device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = False

preprocess = None
emb_model = None
use_openclip = False


try:
    import open_clip
    emb_model, _, preprocess = open_clip.create_model_and_transforms(
        EMB_MODEL_NAME, pretrained=EMB_PRETRAINED, device=device
    )
    emb_model.eval()
    use_openclip = True
    print(f"[INFO] Loaded OpenCLIP {EMB_MODEL_NAME} / {EMB_PRETRAINED}")
except Exception as e:
    print(f"[WARN] OpenCLIP not available ({e}). Falling back to CLIP.")
    try:
        import clip
        emb_model, preprocess = clip.load("ViT-B/32", device=device)
        emb_model.eval()
        use_openclip = False
        print("[INFO] Loaded CLIP ViT-B/32")
    except Exception as e2:
        raise RuntimeError("Không thể nạp bất kỳ CLIP model nào.") from e2

def pil_from_bgr(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

@torch.inference_mode()
def encode_images(img_list_pil):
    # img_list_pil: list[PIL.Image]
    imgs = torch.stack([preprocess(im) for im in img_list_pil]).to(device)
    if use_fp16 and use_openclip:
        imgs = imgs.half()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feats = emb_model.encode_image(imgs)
    else:
        feats = emb_model.encode_image(imgs)
    feats = F.normalize(feats.float(), dim=1)  # L2 norm
    return feats  # shape [N, D]

def encode_patches_batch(pil_list, bs=32):
    outs = []
    for i in range(0, len(pil_list), bs):
        feats = encode_images(pil_list[i:i+bs])  # đã L2-norm
        outs.append(feats)
    return torch.cat(outs, dim=0)

def load_ref_prototype(ref_dir: str) -> torch.Tensor:
    paths = []
    for name in ["img_1.jpg", "img_2.jpg", "img_3.jpg"]:
        p = Path(ref_dir) / name
        if p.exists():
            paths.append(str(p))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh ref trong {ref_dir}")

    imgs_pil = [Image.open(p).convert("RGB") for p in paths]
    feats = encode_images(imgs_pil)            # [n_ref, D], đã L2-normalize
    proto = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)  # [1, D]
    return proto, feats                         # cả prototype lẫn từng ref

def clip_cosine(embs: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    # embs: [N, D], proto: [1, D], cả 2 đã L2-norm -> cosine = dot
    return (embs @ proto.t()).squeeze(1)        # [N]