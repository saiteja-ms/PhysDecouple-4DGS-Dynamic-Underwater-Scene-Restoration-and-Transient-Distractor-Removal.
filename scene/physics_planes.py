"""
scene/physics_planes.py
-----------------------
PhysSplit-4DGS — Akkaynak-revised medium parameterization.

ARCHITECTURE (replaces the old HexPlane):
    β_D(z): 2-term exponential per channel, 12 learnable params
            β_D_c(z) = a_c · exp(b_c·z) + c_c · exp(d_c·z)
            Monotonic decay enforced: a,c > 0 via softplus; b,d < 0 via -softplus
            Output bounded into Jerlov locus per channel via outer sigmoid.

    β_B: single per-scene RGB triple, 3 params
         No depth axis (Akkaynak 2018 Fig 4a: β_B varies weakly with z).
         No time axis (medium constant per scene for typical short clips).
         Bounded sigmoid into Jerlov locus.

    B_∞: single per-scene RGB triple, 3 params
         No depth axis (B_∞ depends on camera vertical depth, not object dist).
         No time axis initially (can add later if videos show clear medium drift).
         Bounded sigmoid into per-channel range from Jerlov.

TOTAL: 18 medium parameters (vs ~18000 in HexPlane). Orders of magnitude harder
to overfit to noise. Physical constraints enforced by parameterization, not by
soft auxiliary losses.

Physical priors:
    JERLOV_PRIORS — per-water-type values for β_D, β_B, B_∞ from literature.
    JERLOV_BOUNDS — per-channel min/max for bounded sigmoid output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Jerlov priors: Akkaynak 2018 / Sea-thru 2019 reference values ──────────
# Values in m⁻¹ for β; in [0,1] for B_∞. Channels [R, G, B].
JERLOV_PRIORS = {
    "I":  {"beta_D": [0.28, 0.11, 0.06],
           "beta_B": [0.03, 0.06, 0.15],
           "B_inf":  [0.55, 0.65, 0.75]},
    "II": {"beta_D": [0.38, 0.19, 0.12],
           "beta_B": [0.06, 0.10, 0.20],
           "B_inf":  [0.50, 0.60, 0.70]},
    "III":{"beta_D": [0.52, 0.32, 0.22],
           "beta_B": [0.11, 0.17, 0.28],
           "B_inf":  [0.45, 0.55, 0.65]},
    "C1": {"beta_D": [0.65, 0.45, 0.35],
           "beta_B": [0.17, 0.23, 0.35],
           "B_inf":  [0.40, 0.50, 0.60]},
    "C3": {"beta_D": [1.05, 0.78, 0.65],
           "beta_B": [0.33, 0.40, 0.55],
           "B_inf":  [0.30, 0.40, 0.50]},
}

# Bounded ranges per channel — used for sigmoid output
JERLOV_BOUNDS = {
    "I":  {"beta_D_min": [0.10, 0.04, 0.02], "beta_D_max": [0.50, 0.20, 0.12],
           "beta_B_min": [0.01, 0.02, 0.05], "beta_B_max": [0.06, 0.12, 0.28],
           "B_inf_min":  [0.30, 0.40, 0.50], "B_inf_max":  [0.70, 0.80, 0.90]},
    "II": {"beta_D_min": [0.22, 0.10, 0.06], "beta_D_max": [0.60, 0.35, 0.22],
           "beta_B_min": [0.03, 0.05, 0.11], "beta_B_max": [0.12, 0.20, 0.35],
           "B_inf_min":  [0.25, 0.35, 0.45], "B_inf_max":  [0.65, 0.75, 0.85]},
    "III":{"beta_D_min": [0.35, 0.18, 0.12], "beta_D_max": [0.80, 0.55, 0.40],
           "beta_B_min": [0.06, 0.10, 0.18], "beta_B_max": [0.22, 0.30, 0.45],
           "B_inf_min":  [0.20, 0.30, 0.40], "B_inf_max":  [0.60, 0.70, 0.80]},
    "C1": {"beta_D_min": [0.45, 0.28, 0.20], "beta_D_max": [0.90, 0.70, 0.60],
           "beta_B_min": [0.10, 0.14, 0.22], "beta_B_max": [0.30, 0.38, 0.55],
           "B_inf_min":  [0.15, 0.25, 0.35], "B_inf_max":  [0.55, 0.65, 0.75]},
    "C3": {"beta_D_min": [0.70, 0.50, 0.40], "beta_D_max": [1.50, 1.20, 1.10],
           "beta_B_min": [0.20, 0.25, 0.35], "beta_B_max": [0.55, 0.65, 0.80],
           "B_inf_min":  [0.10, 0.20, 0.30], "B_inf_max":  [0.50, 0.60, 0.70]},
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _sigmoid_inv(y, lo, hi):
    """
    Inverse of bounded sigmoid: given y in [lo, hi], return raw r such that
    lo + (hi - lo) * sigmoid(r) == y.
    Used to initialize raw parameters so activations match Jerlov priors.
    """
    y = torch.as_tensor(y, dtype=torch.float32)
    lo = torch.as_tensor(lo, dtype=torch.float32)
    hi = torch.as_tensor(hi, dtype=torch.float32)
    t = ((y - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    return torch.log(t / (1 - t))


def _softplus_inv(y):
    """
    Inverse of softplus: given y > 0, return raw r such that softplus(r) == y.
    Used to initialize raw_b, raw_d so b, d match decay rates from Akkaynak.
    """
    y = torch.as_tensor(y, dtype=torch.float32).clamp(min=1e-4)
    return torch.log(torch.expm1(y))


# ── Main PhysicsPlanes module ──────────────────────────────────────────────

class PhysicsPlanes(nn.Module):
    """
    Akkaynak-revised medium parameterization with hard physical constraints.

    Forward path:
        query(depth, time=None) -> dict(beta_D, beta_B, B_inf)
            beta_D : (N, 3) — depth-dependent direct attenuation per query pixel
            beta_B : (3,)   — scene-constant backscatter coefficient
            B_inf  : (3,)   — scene-constant veiling light

    No bilinear interpolation, no grid_sample, no plane storage. Just a small
    parameter set producing bounded, monotonically-decaying β_D(z).
    """

    def __init__(self, water_type="II", depth_res=None, time_res=None):
        """
        Args:
            water_type : str — key into JERLOV_PRIORS, default "II"
            depth_res, time_res : kept for API backwards-compat (unused).
                                  Old HexPlane signature; safe to ignore.
        """
        super().__init__()
        self.water_type = water_type

        prior = JERLOV_PRIORS[water_type]
        bounds = dict(JERLOV_BOUNDS[water_type])

        # B_inf is veiling light: it depends on illumination and camera
        # exposure, NOT on water-type optics — only its spectral ordering
        # (B > G > R) is physical. Per-type level tables forced bad fits
        # (deep blue water has near-zero red veil but type-I min was 0.30,
        # precalib loss 0.15 vs 0.03 on scenes the tables happened to match).
        # Keep beta bounds per water type; make B_inf bounds wide & universal.
        bounds["B_inf_min"] = [0.02, 0.03, 0.05]
        bounds["B_inf_max"] = [0.90, 0.93, 0.95]

        # The spectral-ordering projections in get_betaD_at_depth/get_Binf
        # clamp into per-channel bounds AFTER ordering, so the guarantee
        # (beta_D: R>G>B; beta_B, B_inf: B>G>R) holds only if the bound tables
        # are themselves ordered. Validate so a table edit can't silently
        # break the paper's by-construction claim.
        for k_lo, k_hi, desc in (("beta_D_min", "beta_D_max", "beta_D R>G>B"),):
            lo, hi = bounds[k_lo], bounds[k_hi]
            assert lo[0] > lo[1] > lo[2] and hi[0] > hi[1] > hi[2], \
                f"{desc} bound table not ordered for water type {water_type}"
        for k_lo, k_hi, desc in (("beta_B_min", "beta_B_max", "beta_B B>G>R"),
                                 ("B_inf_min", "B_inf_max", "B_inf B>G>R")):
            lo, hi = bounds[k_lo], bounds[k_hi]
            assert lo[2] > lo[1] > lo[0] and hi[2] > hi[1] > hi[0], \
                f"{desc} bound table not ordered for water type {water_type}"

        # ── Bounds registered as buffers (move to GPU but not learned) ────────
        self.register_buffer(
            "beta_D_min", torch.tensor(bounds["beta_D_min"], dtype=torch.float32))
        self.register_buffer(
            "beta_D_max", torch.tensor(bounds["beta_D_max"], dtype=torch.float32))
        self.register_buffer(
            "beta_B_min", torch.tensor(bounds["beta_B_min"], dtype=torch.float32))
        self.register_buffer(
            "beta_B_max", torch.tensor(bounds["beta_B_max"], dtype=torch.float32))
        self.register_buffer(
            "B_inf_min", torch.tensor(bounds["B_inf_min"], dtype=torch.float32))
        self.register_buffer(
            "B_inf_max", torch.tensor(bounds["B_inf_max"], dtype=torch.float32))

        # ── β_D parameterization: 2-term exponential per channel ──────────────
        # β_D_c(z) = a_c · exp(b_c·z) + c_c · exp(d_c·z)
        # Storage: raw_betaD_coeffs of shape (3, 4) = per channel [raw_a, raw_b, raw_c, raw_d]
        # Activations:
        #   a, c = softplus(raw) → positive
        #   b, d = -softplus(raw) → negative (forces decay)
        #
        # Initialization: for each channel, choose (a, b, c, d) so that at z=0,
        # β_D(0) = a + c matches Jerlov prior, and decay rates are physically
        # reasonable. We use a fast component (b ≈ -0.5) and a slow (d ≈ -0.05).
        # Akkaynak 2018 Fig 5d shows ~50/50 split between fast and slow terms.
        prior_bd = prior["beta_D"]  # [R, G, B] target values at z=0
        raw_betaD = torch.zeros(3, 4)
        for c in range(3):
            target = prior_bd[c]
            a_init = target * 0.5  # fast component half
            c_init = target * 0.5  # slow component half
            b_init = 0.5           # decay rate fast (will be negated)
            d_init = 0.05          # decay rate slow (will be negated)
            raw_betaD[c, 0] = _softplus_inv(a_init)
            raw_betaD[c, 1] = _softplus_inv(b_init)
            raw_betaD[c, 2] = _softplus_inv(c_init)
            raw_betaD[c, 3] = _softplus_inv(d_init)
        self.raw_betaD_coeffs = nn.Parameter(raw_betaD)  # (3, 4)

        # ── β_B parameterization: scene-constant, bounded monotone sigmoid ────
        # β_B is initialized at prior, output ∈ [β_B_min, β_B_max], with
        # hard strict ordering R < G < B.
        betaB_min = torch.tensor(bounds["beta_B_min"], dtype=torch.float32)
        betaB_max = torch.tensor(bounds["beta_B_max"], dtype=torch.float32)
        betaB_prior = torch.tensor(prior["beta_B"], dtype=torch.float32)
        g_lo = torch.maximum(betaB_min[1], betaB_prior[0])
        b_lo = torch.maximum(betaB_min[2], betaB_prior[1])
        raw_betaB = torch.stack([
            _sigmoid_inv(betaB_prior[0], betaB_min[0], betaB_max[0]),
            _sigmoid_inv(betaB_prior[1], g_lo, betaB_max[1]),
            _sigmoid_inv(betaB_prior[2], b_lo, betaB_max[2]),
        ])
        self.raw_betaB = nn.Parameter(raw_betaB)  # (3,)

        # ── B_∞ parameterization: scene-constant, bounded sigmoid ─────────────
        raw_Binf = _sigmoid_inv(
            prior["B_inf"], bounds["B_inf_min"], bounds["B_inf_max"])
        self.raw_Binf = nn.Parameter(raw_Binf)  # (3,)

        # Monocular/SfM scene scale is arbitrary. Query beta on normalized depth,
        # then learn an equivalent optical path length for exp(-beta * d).
        # Max kept at 4.0 so beta_D*d <= ~2.4 and T_D >= ~9%: J must always
        # receive photometric gradient, otherwise it drifts unconstrained in
        # deep regions and backscatter alone absorbs the image.
        self.depth_scale_min = 0.5
        self.depth_scale_max = 4.0
        self.raw_depth_scale = nn.Parameter(
            _sigmoid_inv(3.0, self.depth_scale_min, self.depth_scale_max))

    # ── Activation functions (callable from outside for inspection) ─────────

    def get_betaD_at_depth(self, depth):
        """
        Compute β_D(z) for a tensor of depths.

        Args:
            depth : Tensor of any shape, normalised depth values (or raw — see note)
        Returns:
            β_D : Tensor of shape (*depth.shape, 3)

        Note on depth scale:
            This function expects normalised depth in [0, 1] as the coordinate
            used to query the coefficient field. The returned beta values remain
            attenuation coefficients, so the image-formation exponent must use
            physical/raster depth in the renderer: exp(-beta * depth_physical).
        """
        # Activate raw coefficients with physical constraints
        # raw_betaD_coeffs shape: (3, 4) = per channel [raw_a, raw_b, raw_c, raw_d]
        a = F.softplus(self.raw_betaD_coeffs[:, 0])  # (3,) positive
        b = -F.softplus(self.raw_betaD_coeffs[:, 1])  # (3,) negative
        c = F.softplus(self.raw_betaD_coeffs[:, 2])  # (3,) positive
        d = -F.softplus(self.raw_betaD_coeffs[:, 3])  # (3,) negative

        # Broadcast depth (..., ) to (..., 3) by adding channel axis
        # Result shape will be (..., 3) regardless of input depth shape
        z = depth.unsqueeze(-1)  # (..., 1)

        # 2-term exponential
        # exp clamped to avoid overflow/underflow when raw params drift
        exp_bz = torch.exp((b * z).clamp(min=-20.0, max=20.0))  # (..., 3)
        exp_dz = torch.exp((d * z).clamp(min=-20.0, max=20.0))  # (..., 3)
        beta_raw = a * exp_bz + c * exp_dz  # (..., 3)

        # Bound into Jerlov range. The exponential form is already physical;
        # bounds are safety only.
        beta_D = beta_raw.clamp(min=self.beta_D_min, max=self.beta_D_max)

        # Hard spectral ordering for direct attenuation:
        # red attenuates more strongly than green, green more strongly than
        # blue. This makes beta_D^R > beta_D^G > beta_D^B true by construction,
        # not only through a soft regularizer.
        eps = torch.as_tensor(1e-4, device=beta_D.device, dtype=beta_D.dtype)
        r = beta_D[..., 0]
        g = torch.minimum(beta_D[..., 1], r - eps).clamp(
            min=self.beta_D_min[1], max=self.beta_D_max[1])
        b = torch.minimum(beta_D[..., 2], g - eps).clamp(
            min=self.beta_D_min[2], max=self.beta_D_max[2])
        return torch.stack([r, g, b], dim=-1)

    def get_betaB(self):
        """Return scene-constant β_B, shape (3,), with R < G < B."""
        eps = torch.as_tensor(1e-4, device=self.raw_betaB.device,
                              dtype=self.raw_betaB.dtype)
        r = (self.beta_B_min[0]
             + (self.beta_B_max[0] - self.beta_B_min[0])
             * torch.sigmoid(self.raw_betaB[0]))

        g_lo = torch.maximum(self.beta_B_min[1], r + eps)
        g_span = (self.beta_B_max[1] - g_lo).clamp_min(1e-6)
        g = g_lo + g_span * torch.sigmoid(self.raw_betaB[1])

        b_lo = torch.maximum(self.beta_B_min[2], g + eps)
        b_span = (self.beta_B_max[2] - b_lo).clamp_min(1e-6)
        b = b_lo + b_span * torch.sigmoid(self.raw_betaB[2])
        return torch.stack([r, g, b])

    def get_Binf(self):
        """Return scene-constant B_∞, shape (3,), with R < G < B."""
        raw = (self.B_inf_min
               + (self.B_inf_max - self.B_inf_min)
               * torch.sigmoid(self.raw_Binf))

        # Hard veiling-light ordering: blue > green > red. The Jerlov bounds
        # are compatible with this ordering, so this projection keeps the
        # parameter interpretable throughout training.
        eps = torch.as_tensor(1e-4, device=raw.device, dtype=raw.dtype)
        r = raw[0]
        g = torch.maximum(raw[1], r + eps).clamp(
            min=self.B_inf_min[1], max=self.B_inf_max[1])
        b = torch.maximum(raw[2], g + eps).clamp(
            min=self.B_inf_min[2], max=self.B_inf_max[2])
        return torch.stack([r, g, b])

    @torch.no_grad()
    def set_Binf(self, target):
        """
        Set raw_Binf so get_Binf() reproduces `target` (3,) as closely as the
        bounded ordered construction allows. Inverts get_Binf: R is free in
        its bounds; G and B invert the max(prev+eps, min)..max projection.
        Used to install the open-water measurement of the veiling light.
        """
        t = torch.as_tensor(target, dtype=torch.float32,
                            device=self.raw_Binf.device)
        eps = 1e-4
        r = t[0].clamp(self.B_inf_min[0] + 1e-4, self.B_inf_max[0] - 1e-4)
        self.raw_Binf[0] = _sigmoid_inv(r, self.B_inf_min[0], self.B_inf_max[0])
        g_lo = torch.maximum(self.B_inf_min[1], r + eps)
        g = t[1].clamp(g_lo + 1e-4, self.B_inf_max[1] - 1e-4)
        self.raw_Binf[1] = _sigmoid_inv(g, self.B_inf_min[1], self.B_inf_max[1])
        b_lo = torch.maximum(self.B_inf_min[2], g + eps)
        b = t[2].clamp(b_lo + 1e-4, self.B_inf_max[2] - 1e-4)
        self.raw_Binf[2] = _sigmoid_inv(b, self.B_inf_min[2], self.B_inf_max[2])

    def get_depth_scale(self):
        """Return equivalent optical path length for normalized depth."""
        return (self.depth_scale_min
                + (self.depth_scale_max - self.depth_scale_min)
                * torch.sigmoid(self.raw_depth_scale))

    def load_state_dict(self, state_dict, strict=True):
        if "raw_depth_scale" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["raw_depth_scale"] = self.raw_depth_scale.detach().clone()
        return super().load_state_dict(state_dict, strict=strict)

    # ── Main query API ─────────────────────────────────────────────────────

    def query(self, depth, time=None):
        """
        Query medium parameters at given depth values.

        Args:
            depth : Tensor of shape (N,) — normalised depth values, ideally in [0,1].
                    Pass `depth_raw / scene_depth_max`. The renderer applies
                    the resulting beta values to physical/raster depth.
            time  : Tensor of shape (N,) — IGNORED. Kept for API compat with old code.
                    Future extension: if you add time-varying B_∞ or β_B, use this.

        Returns:
            dict:
                beta_D : (N, 3) — direct attenuation per query depth, per channel
                beta_B : (3,)   — scene-constant backscatter coefficient
                B_inf  : (3,)   — scene-constant veiling light

        Note on tensor shapes for callers:
            - Caller in renderer_uw.py will call query(depth_flat) where depth_flat
              is shape (H*W,) and reshape result back to (H, W, 3).
            - β_B and B_∞ are scalars-per-channel; caller broadcasts as needed.
            - β_D shape matches input depth shape with extra channel dim added.
        """
        if depth.dim() == 0:
            depth = depth.unsqueeze(0)

        beta_D = self.get_betaD_at_depth(depth)  # shape (N, 3) for flat input
        beta_B = self.get_betaB()                 # (3,)
        B_inf = self.get_Binf()                   # (3,)
        depth_scale = self.get_depth_scale()       # scalar

        return dict(beta_D=beta_D, beta_B=beta_B,
                    B_inf=B_inf, depth_scale=depth_scale)

    # ── Compatibility shim: NO-OP methods that old code may still call ─────

    def manifold_loss(self, beta_D=None, beta_B=None):
        """
        DEPRECATED. Returns zero. Hard parameterization bounds replace this.
        Kept as no-op so old call sites don't crash during migration.
        """
        return torch.tensor(0.0, device=self.raw_betaD_coeffs.device)

    def tv_loss(self):
        """
        DEPRECATED. Returns zero. The 2-term exponential is inherently smooth.
        Kept as no-op so old call sites don't crash during migration.
        """
        return torch.tensor(0.0, device=self.raw_betaD_coeffs.device)

    # ── Diagnostic / inspection helpers ─────────────────────────────────────

    def summary(self, depth_samples=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """
        Return a dict with β_D evaluated at given normalised depths plus β_B and B_∞.
        Use for printing diagnostics during/after training.
        """
        with torch.no_grad():
            samples = torch.tensor(depth_samples, device=self.raw_betaD_coeffs.device)
            bd = self.get_betaD_at_depth(samples)  # (N, 3)
            bb = self.get_betaB()
            binf = self.get_Binf()
        return {
            "beta_D_at_depths": dict(zip(depth_samples, bd.cpu().tolist())),
            "beta_B": bb.cpu().tolist(),
            "B_inf":  binf.cpu().tolist(),
        }
