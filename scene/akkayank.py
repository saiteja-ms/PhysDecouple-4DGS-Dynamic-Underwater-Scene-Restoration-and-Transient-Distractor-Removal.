"""
scene/akkayank.py
-----------------
Differentiable Akkaynak-Treibitz underwater image formation model.

Transmission floor at -20 so deep haze can be fully explained by physics.
exp(-20) ≈ 2e-9, effectively zero — lets β·z saturate without numerical issues.

Forward only: J (clean) -> I (underwater/degraded)

Reference:
    Akkaynak & Treibitz, CVPR 2018. "A Revised Underwater Image Formation Model"
"""

import torch
import torch.nn as nn


class AkkayankRenderer(nn.Module):
    """Stateless module — no learnable parameters."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, J, depth, beta_D, beta_B, B_inf):
        """
        Akkaynak forward: I^c = J^c · exp(-β_D^c · d) + B_∞^c · (1 - exp(-β_B^c · d))

        Args:
            J      : tensor (..., 3) — clean colour
            depth  : tensor (...,) — depth (any positive units; β/depth scale must match)
            beta_D : tensor (..., 3) or (3,) — direct attenuation
            beta_B : tensor (..., 3) or (3,) — backscatter attenuation
            B_inf  : tensor (3,) or broadcastable — veiling light

        Returns:
            I : tensor (..., 3), clamped to [0, 1]
        """
        d = depth.unsqueeze(-1).clamp(min=0.0, max=10.0)
        exponent_D = (-beta_D * d).clamp(min=-20.0, max=0.0)
        exponent_B = (-beta_B * d).clamp(min=-20.0, max=0.0)
        trans_D = torch.exp(exponent_D)
        trans_B = torch.exp(exponent_B)
        direct = J * trans_D
        backscatter = B_inf.clamp(0., 1.) * (1.0 - trans_B)
        out = direct + backscatter
        return out.clamp(0.0, 1.0)

    def physics_residual(self, I_rendered, I_akkayank):
        """Per-pixel L1 residual between rasterised colour and Akkaynak prediction."""
        return (I_rendered - I_akkayank).abs().mean(dim=-1)
