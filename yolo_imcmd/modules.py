import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- tiện ích ----
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.cv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.cv(x)))

# ---- Coordinate Attention (CA) theo Hou et al. (CVPR'21) ----
class CoordAtt(nn.Module):
    def __init__(self, c, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c // reduction)
        self.conv1 = nn.Conv2d(c, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, c, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c, 1, 1, 0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                 # (n,c,h,1)
        x_w = self.pool_w(x).permute(0,1,3,2)  # (n,c,1,w)
        y = torch.cat([x_h, x_w], dim=2)     # (n,c,h+1,w or so)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, 1], dim=2)
        x_w = x_w.permute(0,1,3,2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w

# ---- C2f_CA: bỏ residual trong bottleneck, concat + CA (theo paper) ----
class C2f_CA(nn.Module):
    def __init__(self, c, n=1):
        super().__init__()
        self.stem1 = Conv(c, c, 1, 1)
        self.stem2 = Conv(c, c, 3, 1)
        self.blocks = nn.Sequential(*[Conv(c, c, 3, 1) for _ in range(max(1,n-1))])
        self.ca = CoordAtt(c)
        self.proj = Conv(2*c, c, 1, 1)  # concat stem & CA -> project

    def forward(self, x):
        s1 = self.stem1(x)
        y = self.stem2(x)
        y = self.blocks(y)
        y = self.ca(y)
        out = torch.cat([s1, y], dim=1)
        return self.proj(out)

# ---- AMFF: đưa về cùng kích thước, Hadamard fuse + SKA (selective kernel attention) ----
class SKA(nn.Module):
    def __init__(self, c, reduction=8):
        super().__init__()
        self.conv3 = Conv(c, c, 3, 1)
        self.conv5 = Conv(c, c, 5, 1)
        self.fc1 = nn.Conv2d(c, c//reduction, 1, 1, 0)
        self.fc2_3 = nn.Conv2d(c//reduction, c, 1, 1, 0)
        self.fc2_5 = nn.Conv2d(c//reduction, c, 1, 1, 0)

    def forward(self, x):
        u3 = self.conv3(x)
        u5 = self.conv5(x)
        u = u3 + u5
        s = F.adaptive_avg_pool2d(u, 1)
        z = F.silu(self.fc1(s))
        a3 = self.fc2_3(z)
        a5 = self.fc2_5(z)
        w = torch.softmax(torch.cat([a3, a5], dim=1), dim=1)
        w3, w5 = torch.split(w, a3.size(1), dim=1)
        return u3 * w3 + u5 * w5

class AMFF(nn.Module):
    def __init__(self, out_c=128):
        super().__init__()
        self.conv = Conv(3*out_c, out_c, 1, 1)  # sau khi chuẩn hoá kênh
        self.ska  = SKA(out_c)

    def _resize_to(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, feats):
        # feats = [P2, P3, P4] đã được map kênh ngoài (Ultralytics đảm bảo)
        p2, p3, p4 = feats
        target = p2.shape[-2:]
        p3 = self._resize_to(p3, target)
        p4 = self._resize_to(p4, target)
        x = torch.cat([p2, p3, p4], dim=1)
        x = self.conv(x)
        # Hadamard với thông tin kênh khác nhau có thể thêm nếu muốn; ở đây dùng SKA
        return self.ska(x)

# ---- Dynamic Head (lite): scale-aware + spatial-aware + task-aware gọn nhẹ ----
class DynamicHeadLite(nn.Module):
    def __init__(self, nc, c=128):
        super().__init__()
        # Scale-aware: 1x1 -> hard-sigmoid
        self.sa = nn.Sequential(nn.Conv2d(c, c, 1, 1, 0), nn.Hardsigmoid())
        # Spatial-aware: depthwise conv + attention mask
        self.spa = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c), nn.BatchNorm2d(c), nn.SiLU(),
            nn.Conv2d(c, c, 1, 1, 0), nn.Sigmoid()
        )
        # Task-aware: hai nhánh class/reg được gate bằng kênh
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ta_cls = nn.Sequential(nn.Conv2d(c, c//4, 1), nn.SiLU(), nn.Conv2d(c//4, c, 1), nn.Hardsigmoid())
        self.ta_reg = nn.Sequential(nn.Conv2d(c, c//4, 1), nn.SiLU(), nn.Conv2d(c//4, c, 1), nn.Hardsigmoid())

        # heads
        self.cls = nn.Conv2d(c, nc, 1, 1, 0)
        self.box = nn.Conv2d(c, 4,  1, 1, 0)

    def forward(self, x):
        # x: feature map cuối từ AMFF
        s = self.sa(x)
        sp = self.spa(x)
        z = x * s * sp
        g = self.pool(z)
        z_cls = z * self.ta_cls(g)
        z_reg = z * self.ta_reg(g)
        return self.cls(z_cls), self.box(z_reg)
