"""
PCB_YOLO: YOLOv8s written in plain PyTorch so the Vitis AI 2.5 quantizer can trace it.

- Same layer order and parameter names as the Ultralytics model ("model.<i>. ..."), so the
  trained weights load directly.
- Activation is LeakyReLU with slope 26/256, the value the DPU implements in hardware.
- BatchNorm eps = 1e-3 as in Ultralytics (eps is not stored in the checkpoint).
- forward() returns the three raw head outputs (strides 8, 16, 32), each with
  64 box channels (4 sides x 16 DFL bins) followed by 6 class logits.
  Box decoding, sigmoid and NMS run on the Arm CPU of the KV260.

make_leaky_state_dict.py checks this module against the Ultralytics model before quantization.
"""
import torch
import torch.nn as nn

NC = 6
REG_MAX = 16
ALPHA = 26.0 / 256.0
SPPF_CV1_ACT = False     # Ultralytics 8.4: the first 1x1 conv in SPPF has no activation


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.LeakyReLU(ALPHA, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3)
        self.cv2 = Conv(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = self.cv1(x)
        y = [y[:, :self.c], y[:, self.c:]]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, act=SPPF_CV1_ACT)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y0 = self.cv1(x)
        y1 = self.m(y0)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y0, y1, y2, y3], 1))


class Detect(nn.Module):
    def __init__(self, nc=NC, ch=(128, 256, 512)):
        super().__init__()
        c2 = max(16, ch[0] // 4, REG_MAX * 4)    # 64
        c3 = max(ch[0], min(nc, 100))            # 128
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * REG_MAX, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch)

    def forward(self, feats):
        return [torch.cat([self.cv2[i](f), self.cv3[i](f)], 1) for i, f in enumerate(feats)]


class PCB_YOLO(nn.Module):
    """YOLOv8s (width 0.50, depth 0.33), 640 x 640 input, 6 classes."""

    def __init__(self, nc=NC):
        super().__init__()
        up = nn.Upsample(scale_factor=2, mode="nearest")
        self.model = nn.ModuleList([
            Conv(3, 32, 3, 2),          # 0
            Conv(32, 64, 3, 2),         # 1
            C2f(64, 64, 1, True),       # 2
            Conv(64, 128, 3, 2),        # 3
            C2f(128, 128, 2, True),     # 4  -> P3
            Conv(128, 256, 3, 2),       # 5
            C2f(256, 256, 2, True),     # 6  -> P4
            Conv(256, 512, 3, 2),       # 7
            C2f(512, 512, 1, True),     # 8
            SPPF(512, 512, 5),          # 9  -> P5
            up,                         # 10
            nn.Identity(),              # 11 concat with 6
            C2f(768, 256, 1),           # 12
            up,                         # 13
            nn.Identity(),              # 14 concat with 4
            C2f(384, 128, 1),           # 15 -> out P3
            Conv(128, 128, 3, 2),       # 16
            nn.Identity(),              # 17 concat with 12
            C2f(384, 256, 1),           # 18 -> out P4
            Conv(256, 256, 3, 2),       # 19
            nn.Identity(),              # 20 concat with 9
            C2f(768, 512, 1),           # 21 -> out P5
            Detect(nc, (128, 256, 512)),  # 22
        ])

    def forward(self, x):
        m = self.model
        x = m[1](m[0](x))
        x = m[2](x)
        p3 = m[4](m[3](x))
        p4 = m[6](m[5](p3))
        p5 = m[9](m[8](m[7](p4)))
        h12 = m[12](torch.cat([m[10](p5), p4], 1))
        o3 = m[15](torch.cat([m[13](h12), p3], 1))
        o4 = m[18](torch.cat([m[16](o3), h12], 1))
        o5 = m[21](torch.cat([m[19](o4), p5], 1))
        return m[22]([o3, o4, o5])


def load_ultralytics_weights(model, path):
    """Copy weights from an Ultralytics checkpoint (best.pt) into PCB_YOLO. Needs ultralytics installed."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    own = model.state_dict()
    sd = {k: v for k, v in src.items() if k in own}             # drops the fixed DFL conv buffer
    missing = [k for k in own if k not in sd]
    if missing:
        raise KeyError(f"{len(missing)} parameters not found in checkpoint, e.g. {missing[:3]}")
    model.load_state_dict(sd, strict=True)
    return model


if __name__ == "__main__":
    net = PCB_YOLO().eval()
    with torch.no_grad():
        outs = net(torch.zeros(1, 3, 640, 640))
    print([tuple(o.shape) for o in outs], sum(p.numel() for p in net.parameters()) / 1e6, "M params")
