"""
utils/loss_uw.py
----------------
Underwater loss functions for PhysDecouple-4DGS.

The important design constraint is that the auxiliary losses must keep giving
useful gradients. Earlier versions used hard gates for dark-pixel anchors; on
Set4 those gates often returned exactly zero, leaving the clean branch J and the
medium parameters underconstrained. The dark/anchor losses below therefore use
soft gates or top-k fallbacks instead of "not enough pixels -> zero".
"""

import torch
import torch.nn.functional as F


def _zero_like_loss(x):
    return torch.zeros((), device=x.device, dtype=x.dtype)


# ── 1. Conditional Dark Channel Prior on J ─────────────────────────────────

def conditional_dark_channel_loss(J_rendered, dark_threshold=0.18,
                                   fraction_threshold=0.01, patch_size=15,
                                   I_observed=None, min_weight=0.20):
    """
    Soft-gated DCP on J.

    For natural images with dark objects (rocks, shadows, coral), the dark
    channel — min over RGB channels, then min over local patch — should be
    near zero. If J is absorbing the water cast, all channels stay elevated
    and dark channel is high, costing loss.

    For bright sandy-bottom scenes with little dark content, this prior can be
    too strong. Instead of disabling it completely, we scale it by a soft scene
    weight and keep a small floor so the loss never becomes permanently dead.

    Args:
        J_rendered         : (3, H, W) — clean rendered scene
        dark_threshold     : float — pixel considered "dark" if min_channel < this
        fraction_threshold : float — apply loss only if >fraction% pixels are dark
        patch_size         : int — local patch size for spatial min
        I_observed         : (3, H, W) — observed ground-truth image

    Returns:
        scalar loss tensor. The loss keeps a non-zero gradient even if the
        observed image has few pixels below the dark threshold.
    """
    # Check if scene has enough dark content using I_observed if available to avoid self-gating
    with torch.no_grad():
        ref_img = I_observed if I_observed is not None else J_rendered
        per_pixel_min = ref_img.min(dim=0).values  # (H, W)
        thresh = dark_threshold
        dark_fraction = (per_pixel_min < thresh).float().mean()
        if fraction_threshold <= 0:
            scene_weight = torch.tensor(1.0, device=J_rendered.device,
                                        dtype=J_rendered.dtype)
        else:
            scene_weight = (dark_fraction / fraction_threshold).clamp(
                min=float(min_weight), max=1.0)

    # Scene has dark content — apply DCP
    J = J_rendered.unsqueeze(0)  # (1, 3, H, W)
    dark_per_channel = J.min(dim=1, keepdim=True).values  # (1, 1, H, W)
    # Local minimum via -maxpool(-x) = erosion
    dark = -F.max_pool2d(
        -dark_per_channel,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2)
    return scene_weight * dark.mean()


def _select_dark_depth_pixels(I_observed, depth=None, dark_threshold=0.25,
                              min_depth=0.25, min_pixels=256,
                              top_fraction=0.02):
    """
    Select informative pixels for dark/deep underwater anchors.

    This helper intentionally has no "return empty" branch. If strict dark/deep
    criteria fail, it falls back to the best-scoring pixels. The estimate is
    detached in callers; gradients only flow to model predictions.
    """
    device = I_observed.device
    I = I_observed.detach().clamp(0.0, 1.0)
    _, H, W = I.shape
    n_pix = H * W

    I_min = I.min(dim=0).values
    I_max = I.max(dim=0).values
    dark_conf = ((dark_threshold - I_min) / max(dark_threshold, 1e-6)).clamp(0.0, 1.0)
    sat_conf = ((0.98 - I_max) / 0.98).clamp(0.0, 1.0)

    if depth is not None:
        d = depth.detach()
        if d.dim() == 3:
            d = d.squeeze(0)
        d = d.clamp(0.0, 1.0)
        depth_conf = ((d - min_depth) / max(1.0 - min_depth, 1e-6)).clamp(0.0, 1.0)
    else:
        depth_conf = torch.ones((H, W), device=device, dtype=I.dtype)

    # Multiplicative soft score with a small baseline so top-k remains useful
    # even when a scene has no textbook "dark and deep" pixels.
    score = (0.10 + dark_conf) * (0.10 + depth_conf) * (0.10 + sat_conf)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)

    k = max(int(min_pixels), int(top_fraction * n_pix))
    k = min(max(k, 1), n_pix)
    top_score, top_idx = torch.topk(score, k=k, largest=True, sorted=False)
    weights = top_score / top_score.mean().clamp_min(1e-6)
    return top_idx, weights.detach()


# ── 2. Dark-pixel B_∞ anchor (Sea-thru 2019 Section 4.3) ───────────────────

def dark_pixel_binf_anchor(I_observed, J_rendered, depth, B_inf_predicted,
                            dark_threshold=0.25, min_pixels=256, percentile=0.9,
                            min_depth=0.25):
    """
    Direct B_∞ supervision from dark/deep pixels in the observed image.

    Physical reasoning (Akkaynak Sea-thru 2019 §4.3): in pixels where J ≈ 0
    (truly dark scene content) at moderate-to-large depth, the observed
    intensity I approaches B_∞ · (1 - exp(-β_B · z)) ≈ B_∞ as z grows.

    So dark pixels give us direct B_∞ measurements. We take the upper envelope
    (high percentile, not max, for robustness) of observed I at dark pixels
    as our B_∞ estimate, and anchor the model's B_∞ to this.

    Args:
        I_observed     : (3, H, W) — observed degraded image
        J_rendered     : (3, H, W) — clean rendered scene (current model output)
        depth          : (H, W) — rasterized depth map
        B_inf_predicted: (3,) — current B_∞ from physics_planes.get_Binf()
        dark_threshold : float — I min < this counts as dark
        min_pixels     : int — minimum dark pixels needed to compute estimate
        percentile     : float — use this percentile of I at dark pixels
        min_depth      : float — minimum normalized depth to use for veiling light anchor

    Returns:
        scalar loss tensor. Uses a top-k fallback, so it does not silently
        return zero when strict dark/deep masks are unavailable.
    """
    del J_rendered
    with torch.no_grad():
        idx, _ = _select_dark_depth_pixels(
            I_observed, depth=depth, dark_threshold=dark_threshold,
            min_depth=min_depth, min_pixels=min_pixels)
        I_dark = I_observed.detach().permute(1, 2, 0).reshape(-1, 3)[idx]
        sorted_I, _ = torch.sort(I_dark, dim=0)
        q = min(max(float(percentile), 0.0), 1.0)
        q_idx = int(q * (I_dark.shape[0] - 1))
        B_inf_estimated = sorted_I[q_idx].clamp(0.0, 1.0)

    # Anchor predicted B_∞ to the data-driven estimate
    # .detach() above ensures gradient only flows through B_inf_predicted
    return (B_inf_predicted - B_inf_estimated).abs().mean()


# ── 3. Gaussian Splashing / WaterClear-style forward-restoration terms ─────

def depth_weighted_l1_loss(I_predicted, I_observed, depth_norm,
                           gamma=1.0, eps=1e-6):
    """
    Photometric loss with slightly higher weight at larger water column depth.

    This keeps the forward model honest in far pixels where water effects are
    strongest, without replacing the normal full-image L1/SSIM objective.
    """
    if depth_norm is None:
        return F.l1_loss(I_predicted, I_observed)
    d = depth_norm.detach()
    if d.dim() == 3:
        d = d.squeeze(0)
    d = d.clamp(0.0, 1.0)
    weights = (1.0 + gamma * d).unsqueeze(0)
    weights = weights / weights.mean().clamp_min(eps)
    return (weights * (I_predicted - I_observed).abs()).mean()


def tone_mapped_l1_loss(I_predicted, I_observed, mu=10.0, eps=1e-6):
    """
    Log/tone-mapped L1. It gives dark regions a useful gradient instead of
    letting bright veiling-light pixels dominate the restoration signal.
    """
    mu = max(float(mu), eps)
    denom = torch.log1p(torch.tensor(mu, device=I_predicted.device,
                                     dtype=I_predicted.dtype))
    pred = torch.log1p(mu * I_predicted.clamp(0.0, 1.0)) / denom
    obs = torch.log1p(mu * I_observed.clamp(0.0, 1.0)) / denom
    return (pred - obs).abs().mean()


def depth_edge_confidence(depth_norm, sharpness=6.0, eps=1e-6):
    """
    Deterministic uncertainty proxy inspired by uncertainty-aware underwater
    reconstruction: pixels at depth discontinuities are less reliable for
    photometric supervision, because splat compositing/depth ambiguity is high.
    """
    if depth_norm is None:
        return None
    d = depth_norm.detach()
    if d.dim() == 3:
        d = d.squeeze(0)
    d = d.clamp(0.0, 1.0)
    gx = F.pad((d[:, 1:] - d[:, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((d[1:, :] - d[:-1, :]).abs(), (0, 0, 0, 1))
    grad = gx + gy
    grad = grad / grad.mean().clamp_min(eps)
    conf = torch.exp(-sharpness * grad).clamp(0.05, 1.0)
    return conf / conf.mean().clamp_min(eps)


def uncertainty_weighted_l1_loss(I_predicted, I_observed, depth_norm,
                                 sharpness=6.0):
    """
    Extra low-weight photometric term with a fixed confidence map. Confidence is
    detached so the model cannot flatten depth merely to reduce its loss.
    """
    conf = depth_edge_confidence(depth_norm, sharpness=sharpness)
    if conf is None:
        return F.l1_loss(I_predicted, I_observed)
    return (conf.unsqueeze(0) * (I_predicted - I_observed).abs()).mean()


def exposure_loss(J_rendered, upper=0.92):
    """
    Prevent the restored clean branch J from becoming an over-bright escape
    solution. Only the upper tail is penalized so genuine dark content is safe.
    """
    return F.relu(J_rendered.clamp(0.0, 1.0) - upper).pow(2).mean()


def gray_world_loss(J_rendered, blue_margin=0.08, green_margin=0.05,
                    red_margin=0.20, warm_margin=0.10):
    """
    Weak clean-color prior for restored J — two-sided.

    A full gray-world prior is too strong for reef/rock scenes because the
    clean scene is not guaranteed to have equal channel means. This version
    penalizes (a) excessive residual blue/green dominance, (b) extreme red
    boosts, and (c) blue collapse (warm over-correction). (c) is essential:
    B_inf over-subtraction drives J yellow/orange (v14: B-R fell to -0.27 on
    set5/set8) and the one-sided form was structurally blind to it — nothing
    in the loss set opposed the shift.
    """
    means = J_rendered.clamp(0.0, 1.0).reshape(3, -1).mean(dim=1)
    r, g, b = means[0], means[1], means[2]
    loss_blue = F.relu(b - r - blue_margin) + F.relu(b - g - blue_margin)
    loss_green = F.relu(g - r - green_margin)
    loss_red = F.relu(r - torch.maximum(g, b) - red_margin)
    loss_warm = F.relu(r - b - warm_margin) + F.relu(g - b - warm_margin)
    return (loss_blue + loss_green + 0.5 * loss_red + loss_warm) / 5.0


def depth_color_decorrelation_loss(J_rendered, depth_norm, fg_mask=None,
                                    eps=1e-6, min_pixels=512):
    """
    Scene- and water-type-agnostic restoration prior:
    restored radiance must be statistically independent of camera range.

    The medium is the only range-dependent process in the formation model, so
    after correct removal the log-chroma of J must show ~zero correlation with
    depth. Any residual correlation IS unremoved (or over-removed) water —
    this single constraint covers yellow/red/green/blue shifts on any dataset
    without hue-specific margins, and it constrains exactly the far-field
    pixels where T_D -> 0 leaves J without photometric gradient. (It is also
    Sea-thru's own validation criterion: color constancy across distance.)
    """
    J = J_rendered.clamp(1e-3, 1.0)
    log_rb = torch.log(J[0]) - torch.log(J[2])
    log_gb = torch.log(J[1]) - torch.log(J[2])
    d = depth_norm
    if fg_mask is None:
        m = torch.ones_like(d, dtype=torch.bool)
    else:
        m = fg_mask
    if m.sum() < min_pixels:
        return torch.zeros((), device=J.device, dtype=J.dtype)
    dv = d[m]
    dv = dv - dv.mean()
    d_std = dv.square().mean().sqrt() + eps
    loss = torch.zeros((), device=J.device, dtype=J.dtype)
    # Dead zone: weak chroma-range correlation occurs naturally (materials
    # are not uniformly distributed over distance); only the strong,
    # systematic correlation that the medium induces is penalized.
    margin = 0.10
    for x in (log_rb, log_gb):
        xv = x[m]
        xv = xv - xv.mean()
        corr = (xv * dv).mean() / (xv.square().mean().sqrt() * d_std + eps)
        loss = loss + torch.relu(corr.abs() - margin)
    return 0.5 * loss


def saturation_loss(J_rendered, threshold=0.98):
    """
    Prevent the restored branch from escaping through clipped/highly saturated
    colors while still allowing normal vivid texture.
    """
    J = J_rendered.clamp(0.0, 1.0)
    max_c = J.max(dim=0).values
    min_c = J.min(dim=0).values
    sat = max_c - min_c
    return F.relu(sat - threshold).pow(2).mean()


def edge_aware_depth_tv_loss(depth_norm, image, eps=1e-6):
    """
    Smooth depth in textureless water/flat surfaces while preserving image edges.
    This fights floaters without blurring object boundaries.
    """
    if depth_norm is None:
        return torch.zeros((), device=image.device, dtype=image.dtype)
    d = depth_norm
    if d.dim() == 3:
        d = d.squeeze(0)
    d = d.clamp(0.0, 1.0)
    img = image.clamp(0.0, 1.0)

    dx = (d[:, 1:] - d[:, :-1]).abs()
    dy = (d[1:, :] - d[:-1, :]).abs()
    img_dx = (img[:, :, 1:] - img[:, :, :-1]).abs().mean(dim=0)
    img_dy = (img[:, 1:, :] - img[:, :-1, :]).abs().mean(dim=0)
    wx = torch.exp(-10.0 * img_dx.detach())
    wy = torch.exp(-10.0 * img_dy.detach())
    return (wx * dx).mean() + (wy * dy).mean()


def spectral_ordering_loss(beta_D, beta_B, B_inf, margin=1e-4):
    """
    Soft underwater spectral prior:
        direct attenuation coefficients: R > G > B
        backscatter coefficients: R < G < B
        veiling light: B > G > R

    Bounds in PhysicsPlanes do most of the work; this term is intentionally
    soft and should be kept low-weight.
    """
    ref = B_inf if B_inf is not None else beta_B
    ref = ref if ref is not None else beta_D
    if ref is None:
        return torch.tensor(0.0)
    loss = _zero_like_loss(ref)
    if beta_D is not None:
        bd = beta_D.reshape(-1, 3).mean(dim=0)
        loss = loss + F.relu(bd[1] - bd[0] + margin)
        loss = loss + F.relu(bd[2] - bd[1] + margin)
    if beta_B is not None:
        bb = beta_B.reshape(-1, 3).mean(dim=0)
        loss = loss + F.relu(bb[0] - bb[1] + margin)
        loss = loss + F.relu(bb[1] - bb[2] + margin)
    if B_inf is not None:
        bi = B_inf.reshape(-1, 3).mean(dim=0)
        loss = loss + F.relu(bi[0] - bi[1] + margin)
        loss = loss + F.relu(bi[1] - bi[2] + margin)
    return loss / 6.0


def dark_pixel_backscatter_loss(I_observed, J_rendered, backscatter_pred,
                                depth_norm=None, dark_threshold=0.25,
                                min_depth=0.25, min_pixels=256):
    """
    In dark/deep pixels, observed intensity is mostly backscatter. This
    supervises the predicted backscatter image, not only global B_inf.

    Uses soft top-k selection so the loss remains active on scenes where the
    old hard mask found too few pixels.
    """
    del J_rendered
    with torch.no_grad():
        idx, weights = _select_dark_depth_pixels(
            I_observed, depth=depth_norm, dark_threshold=dark_threshold,
            min_depth=min_depth, min_pixels=min_pixels)

    pred = backscatter_pred.permute(1, 2, 0).reshape(-1, 3)[idx]
    obs = I_observed.permute(1, 2, 0).reshape(-1, 3)[idx]
    return (weights.unsqueeze(-1) * (pred - obs).abs()).mean()


# ── 4. Physics effect loss — prevents degenerate J ≈ I_GT escape ────────────

def physics_ordering_loss(J_rendered, I_degraded, depth_norm=None,
                           min_red_atten=0.03, min_blue_add=0.02,
                           min_depth=0.25):
    """
    Force the physics forward model to have a measurable effect.

    Two complementary sub-terms:

    (a) Minimum-effect floor  — catches the degenerate I_deg ≈ J case.
        The water column MUST attenuate red by at least min_red_atten and
        add at least min_blue_add of blue veiling, in the channel means.
        When the optimizer collapses the physics to near-identity both terms
        are non-zero and provide gradients that push J and I_deg apart.

    (b) Ordering constraint — catches the inverted case where I_R > J_R or
        I_B < J_B (physics running backwards).

    Combined, they form a tight physical prior:
        J_R − I_R > min_red_atten   AND   I_B − J_B > min_blue_add
                                    AND   I_R < J_R   AND   I_B > J_B

    Gradients flow through both J_rendered (Gaussian SH) and I_degraded
    (physics_planes β_D, β_B, B_∞).
    """
    weights = None
    if depth_norm is not None:
        with torch.no_grad():
            d = depth_norm.detach()
            if d.dim() == 3:
                d = d.squeeze(0)
            weights = ((d - min_depth) / max(1e-6, 1.0 - min_depth)).clamp(0, 1)
            if weights.sum() < 1.0:
                weights = None

    def channel_mean(x, c):
        if weights is None:
            return x[c].mean()
        return (x[c] * weights).sum() / weights.sum().clamp_min(1e-6)

    j_r = channel_mean(J_rendered, 0)
    j_b = channel_mean(J_rendered, 2)
    i_r = channel_mean(I_degraded, 0)
    i_b = channel_mean(I_degraded, 2)

    # (a) Minimum effect: penalise when the effect is too small
    # Fires when I_deg ≈ J (near-identity physics — the degenerate escape)
    red_floor  = F.relu(min_red_atten - (j_r - i_r))  # J_R − I_R < floor
    blue_floor = F.relu(min_blue_add  - (i_b - j_b))  # I_B − J_B < floor

    # (b) Ordering: penalise physically backwards configurations
    red_order  = F.relu(i_r - j_r + 1e-4)   # I_R must be < J_R
    blue_order = F.relu(j_b - i_b + 1e-4)   # I_B must be > J_B

    return red_floor + blue_floor + red_order + blue_order


# ── 5. Spatial Residual Sigma Assignment — RGSSA (novel) ───────────────────

def spatial_residual_sigma_loss(sigma_render, I_degraded, I_observed,
                                deadband=0.015, quantile=0.95):
    """
    Novel: align the per-pixel σ render to the per-pixel photometric residual.

    Motivation
    ----------
    The deformation-norm branch of the transient loss misses objects that move
    smoothly or whose motion is absorbed by the deformation network (e.g. slowly
    drifting fish, lazy-floating particles).  On Set4 all σ values collapsed to
    0.05 — zero Gaussians were ever classified as transient.

    RGSSA (Residual-Guided Spatial Sigma Assignment) uses the forward model's
    own prediction error as a complementary spatial signal:

        R(p) = |I_degraded(p) − I_observed(p)|   (per-pixel, mean over channels)

    Pixels where the Akkaynak forward model cannot explain the observation
    (high R) are occluded by content not represented in the static scene —
    i.e. transients.  We render a per-pixel mean-σ image and push it toward a
    normalised version of R.

    No external transient labels are required.  The signal is self-supervised:
    the Gaussians learn to explain the water medium, and whatever they cannot
    explain is flagged as transient via σ.

    This is complementary to deformation-based detection: RGSSA fires on any
    residual regardless of cause; the deformation branch fires specifically on
    moving geometry.

    Args:
        sigma_render : (1, H, W) or (H, W) — alpha-weighted mean σ per pixel
                       from the renderer (key "sigma_render" in render_pkg)
        I_degraded   : (3, H, W) — physics-forward rendered degraded image
        I_observed   : (3, H, W) — ground-truth observed image
    """
    with torch.no_grad():
        residual = (I_degraded.detach() - I_observed.detach()).abs()
        R = residual.mean(dim=0)                            # (H, W)
        scale = torch.quantile(R.flatten(), quantile).clamp_min(deadband + 1e-5)
        R_norm = ((R - deadband).clamp_min(0.0)
                  / (scale - deadband).clamp_min(1e-5))
        R_norm = R_norm.clamp(0.0, 1.0)

    sig = sigma_render
    if sig.dim() == 3:
        sig = sig.squeeze(0)                                # (H, W)
    sig = sig.clamp(0.0, 1.0)
    return F.mse_loss(sig, R_norm)


# ── 6. Transient loss (σ machinery) ────────────────────────────────────────

def transient_loss(sigma_hat, deform_norm, I_observed, I_predicted,
                    tau_motion=0.05, lambda_motion=1.0, lambda_phys=0.0):
    """
    Push σ high for Gaussians that are moving.

    L_motion:   m_i · (1 - σ_i)
        m_i = tanh(deform_norm_i / τ_motion) — high for moving Gaussians.
        Static Gaussians with high σ pay no penalty.
        Moving Gaussians with low σ are penalised.

    The residual term is deliberately tiny because it is scene-level rather
    than spatially assigned to individual Gaussians. It prevents a fully dead
    branch, while the sigma budget keeps it from marking everything transient.

    Args:
        sigma_hat   : (N,) — activated σ scores
        deform_norm : (N,) — L2 norm of deformation per Gaussian (EMA in train.py)
        I_observed  : (H, W, 3) or (3, H, W) — GT (will be detached anyway)
        I_predicted : (H, W, 3) or (3, H, W) — model's Akkaynak forward output
        tau_motion  : float — deformation scale for tanh normalization
        lambda_motion, lambda_phys : float — sub-term weights

    Returns:
        (loss_scalar, dict of components)
    """
    deform_norm = deform_norm.clamp(0.0, 100.0)
    tanh_input = (deform_norm / (tau_motion + 1e-6)).clamp(-10.0, 10.0)
    m_i = torch.tanh(tanh_input).clamp(0.0, 1.0)

    if not torch.isfinite(m_i).all():
        m_i = torch.zeros_like(m_i)

    # Absolute deformation can be tiny in stabilized 4DGS runs. Add a relative
    # motion cue so the top-moving Gaussians still receive transient pressure
    # instead of letting sparsity drive every σ to the floor.
    with torch.no_grad():
        finite_deform = deform_norm[torch.isfinite(deform_norm)]
        if finite_deform.numel() > 8 and finite_deform.max() > 1e-8:
            q50 = torch.quantile(finite_deform, 0.50)
            q90 = torch.quantile(finite_deform, 0.90)
            scale = (q90 - q50).clamp_min(1e-8)
            rel_motion = ((deform_norm - q50) / scale).clamp(0.0, 1.0)
            m_i = torch.maximum(m_i, rel_motion)

    # m_i is a supervision signal — detach so gradients don't flow to deformation net
    l_motion = (m_i.detach() * (1.0 - sigma_hat)).mean()

    diff = (I_predicted.detach() - I_observed.detach()).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0).clamp(0.0, 0.5)
    r_bar = diff.mean()
    l_residual = r_bar * (1.0 - sigma_hat).mean()

    loss = lambda_motion * l_motion + lambda_phys * l_residual

    if not torch.isfinite(loss):
        return (torch.tensor(0.0, device=sigma_hat.device, requires_grad=True),
                dict(l_motion=torch.tensor(0.0), l_residual=torch.tensor(0.0)))

    return loss, dict(l_motion=l_motion, l_residual=l_residual)


# ── 5. Opacity sparsity loss ───────────────────────────────────────────────

def opacity_sparsity_loss(sigma_hat, opacity=None, target_mean=0.08,
                          lower_bound=0.03, upper_bound=0.20,
                          binarize_weight=0.05):
    """
    Keep transient σ selective without collapsing it to all-zero.

    A pure mean(σ) penalty killed the transient branch on Set4: all values fell
    to the lower clamp and stayed there. This loss uses a small budget band
    instead. It penalizes too many transient Gaussians, but also penalizes a
    completely inactive branch. The optional binarization term encourages
    separation once motion/residual terms create variation.

    Returns:
        scalar loss
    """
    del opacity
    mean_sigma = sigma_hat.mean()
    target = torch.as_tensor(target_mean, device=sigma_hat.device,
                             dtype=sigma_hat.dtype)
    lower = torch.as_tensor(lower_bound, device=sigma_hat.device,
                            dtype=sigma_hat.dtype)
    upper = torch.as_tensor(upper_bound, device=sigma_hat.device,
                            dtype=sigma_hat.dtype)
    budget = F.relu(lower - mean_sigma).pow(2) + F.relu(mean_sigma - upper).pow(2)
    target_pull = 0.10 * (mean_sigma - target).pow(2)
    binarize = (sigma_hat * (1.0 - sigma_hat)).mean()
    return budget + target_pull + binarize_weight * binarize


# ── 6. Combined wrapper (for convenience; train.py can also call directly) ─

def uw_total_loss(sigma_hat, deform_norm, opacity,
                   I_observed, I_predicted,
                   iteration, args):
    """
    Compute the σ-machinery losses (transient + sparsity) with warmup schedule.

    Note: photometric, dark channel, and B_∞ anchor losses are computed
    directly in train.py because they need access to render outputs and
    physics_planes that aren't easily passed here.

    Args:
        sigma_hat   : (N,) get_sigma from UnderwaterGaussianModel
        deform_norm : (N,) EMA of deformation_accum norm
        opacity     : (N,) get_opacity().squeeze(-1)
        I_observed  : tensor — observed GT for residual signal
        I_predicted : tensor — model's Akkaynak forward output
        iteration   : int — current step (for warmup)
        args        : argparse Namespace with uw_* fields

    Returns:
        (total_loss, components_dict)
    """
    lam_t = getattr(args, "uw_lambda_transient", 0.03)
    lam_s = getattr(args, "uw_lambda_sparse", 0.0002)
    tau_m = getattr(args, "uw_tau_motion", 0.05)
    lam_res = getattr(args, "uw_lambda_transient_residual", 0.02)

    l_trans, trans_comps = transient_loss(
        sigma_hat, deform_norm, I_observed, I_predicted,
        tau_motion=tau_m, lambda_phys=lam_res)
    l_sparse = opacity_sparsity_loss(
        sigma_hat, opacity,
        target_mean=getattr(args, "uw_sigma_target_mean", 0.08),
        lower_bound=getattr(args, "uw_sigma_lower_bound", 0.03),
        upper_bound=getattr(args, "uw_sigma_upper_bound", 0.20),
        binarize_weight=getattr(args, "uw_sigma_binarize_weight", 0.05))

    # NaN guards
    l_trans = torch.nan_to_num(l_trans, nan=0.0, posinf=0.0).clamp(0.0, 10.0)
    l_sparse = torch.nan_to_num(l_sparse, nan=0.0, posinf=0.0).clamp(0.0, 10.0)

    total = lam_t * l_trans + lam_s * l_sparse
    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

    components = dict(
        uw_total=total,
        uw_trans=l_trans,
        uw_sparse=l_sparse,
        uw_warmup_w=torch.tensor(1.0, device=sigma_hat.device),
        **{f"trans_{k}": v for k, v in trans_comps.items()},
    )
    return total, components
