"""Noam learning-rate schedule from Vaswani et al. 2017 §5.3.

    lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))

Increases linearly during warmup, then decays as 1/sqrt(step).  The peak LR is
reached at `step = warmup_steps` and equals `d_model^(-0.5) · warmup^(-0.5)`.

Wrap an existing optimizer; call `step()` once per training step.
"""
from __future__ import annotations

import torch


class NoamLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        scale = d_model ** -0.5

        def fn(step: int) -> float:
            # +1 because LambdaLR steps start at 0 internally.
            s = max(step + 1, 1)
            return scale * min(s ** -0.5, s * (warmup_steps ** -1.5))

        super().__init__(optimizer, lr_lambda=fn, last_epoch=last_epoch)
