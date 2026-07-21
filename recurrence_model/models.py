from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig
from .basic_model import CausalTransformerBaseline
from .depth_recurrence_model import DepthRatioGt1Model, DepthRatio1Model, DepthRatioLt1Model
from .depth_step_recurrence_model import DepthStepRatioGt1Model, DepthStepRatio1Model, DepthStepRatioLt1Model
from .step_recurrence_model import (
    BlockRecurrentTransformerModel,
    DeltaProductModel,
    MambaRatio1Model,
)


def make_model(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind == "baseline":
        return CausalTransformerBaseline(cfg)
    if kind in {"ratiogt1", "depthRatiogt1"}:
        return DepthRatioGt1Model(cfg)
    if kind == "depthRatio1":
        return DepthRatio1Model(cfg)
    if kind == "depthRatiolt1":
        return DepthRatioLt1Model(cfg)
    if kind in {"depthstep_gt1", "depthstepRatiogt1"}:
        return DepthStepRatioGt1Model(cfg)
    if kind in {"ratio1", "depthstepRatio1"}:
        return DepthStepRatio1Model(cfg)
    if kind in {"ratiolt1", "depthstepRatiolt1"}:
        return DepthStepRatioLt1Model(cfg)
    if kind == "deltaproduct":
        return DeltaProductModel(cfg)
    if kind == "brt":
        return BlockRecurrentTransformerModel(cfg)
    if kind == "mamba":
        return MambaRatio1Model(cfg)
    raise ValueError(f"Unknown model kind: {kind}")
