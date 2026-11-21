"""Policy/value ResNet for Chaturaji (see bot/PLAN.md section 1.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import NUM_ACTIONS, NUM_PLANES


@dataclass
class NetConfig:
    channels: int = 128
    blocks: int = 10
    value_hidden: int = 256


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ChaturajiNet(nn.Module):
    """Input (B, NUM_PLANES, 8, 8).

    Returns:
      policy_logits (B, 4096): action = from_square * 64 + to_square (canonical coords)
      value_logits  (B, 4):    top-score probability per relative seat (me, next, across, prev)
      score_pred    (B, 4):    predicted final score margin per relative seat
    """

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        ch = self.cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(self.cfg.blocks)])
        # policy: channel = destination square, spatial position = origin square
        self.policy = nn.Sequential(
            nn.Conv2d(ch, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1))
        self.value_conv = nn.Sequential(nn.Conv2d(ch, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(32 * 64, self.cfg.value_hidden), nn.ReLU())
        self.value_out = nn.Linear(self.cfg.value_hidden, 4)
        self.score_out = nn.Linear(self.cfg.value_hidden, 4)

    def forward(self, x):
        h = self.trunk(self.stem(x))
        p = self.policy(h)                                  # (B, 64[to], 8, 8[from])
        p = p.permute(0, 2, 3, 1).reshape(x.shape[0], NUM_ACTIONS)  # from*64 + to
        v = self.value_fc(self.value_conv(h).flatten(1))
        return p, self.value_out(v), self.score_out(v)


def save_checkpoint(path, model: ChaturajiNet, optimizer=None, step: int = 0, extra=None):
    torch.save({
        "config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "step": step,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu") -> tuple[ChaturajiNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ChaturajiNet(NetConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
