"""
PhysDecouple-4DGS underwater training config.

Use CLI for:
    -s / --source_path
    -m / --model_path
    --stage
    --water_type

This file sets optimization, deformation, and underwater loss defaults.

v13 (structural fixes, see v12 post-mortem):
    - depth is DETACHED inside the physics composition (renderer_uw.py);
      the v11/v12 collapse was the optimizer sculpting per-pixel depth
      through exp(-beta*d) as a free color knob ("depth painting").
    - physics anchor losses (binf/dark/backscatter-dark/ordering) are no
      longer gated by the photo guard (train.py uw_anchor_w).
    - fine stage blends the physics composition in over uw_warmup_iters
      instead of switching the photometric target in one step.
    - opacity_reset_interval 3000 -> 300000 (upstream 4DGS HyperNeRF value).
    - physics_planes depth_scale_max 8 -> 4 (T_D floor ~9%).

v14 (medium-first adaptation, see v13 set5 post-mortem):
    - medium pre-calibration at fine start: fit the 18 physics params to
      (J_coarse, GT) pairs before any fine photometric step. Without it the
      Jerlov priors are far from the data and the fast scene variables absorb
      the mismatch (orange J, blurred geometry) before the medium can move.
    - uw_lr_planes 5e-4 -> 5e-3, physics grad clip 0.1 -> 1.0.
    - anchors active from fine iter 0 (ramp over 500), incl. mono-depth/TV.
    - >300k emergency prune now respects uw_disable_fine_pruning.

v15 (v14 post-mortem on set4/5/8 renders):
    - sigma-gated robust photometric + transient opacity decay: transients
      were reconstructed as view-overfit SH3 Gaussians (magenta/green blobs
      from novel views, temporal pop-in) — now they are downweighted in the
      photometric loss and faded out, so masked renders have no holes.
    - yellow-shift fix: gray-world moved to anchor group at 0.02 (was
      0.002*effective ~ off; v14 set5 J had B-R = -0.27 vs GT +0.05),
      uw_binf_percentile 0.9 -> 0.6, uw_lambda_dark 0.05 -> 0.03.
    - capacity: densify cap 350k -> uw_max_points 700k (v14 froze at the cap
      all fine: zero fine densification), opacity-only pruning in UW fine,
      deformation net to upstream 4DGS size (net_width 128, multires [1,2,4]).
"""


class ModelParams:
    white_background = False
    sh_degree = 3
    images = "images"
    resolution = -1
    data_device = "cuda"
    eval = True
    render_process = False
    add_points = False
    extension = ".png"
    llffhold = 8


class OptimizationParams:
    # Main schedule
    iterations = 25000
    coarse_iterations = 4000
    batch_size = 1

    # 4DGS optimization
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 25000
    deformation_lr_init = 0.00005
    deformation_lr_final = 0.000005
    deformation_lr_delay_mult = 0.01
    grid_lr_init = 0.0016
    grid_lr_final = 0.000016
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    lambda_lpips = 0

    # Densification/pruning
    densification_interval = 100
    # Never reset opacity: upstream 4DGS HyperNeRF uses 300000 (off). With
    # 3000 the reset fired at coarse iter 3000, mass-pruning low-contrast
    # underwater Gaussians 1000 iters before the fine handoff (v12 post-mortem:
    # coarse l1 0.09 -> 0.15 after reset, never recovered).
    opacity_reset_interval = 300000
    densify_from_iter = 500
    densify_until_iter = 12000
    densify_grad_threshold_coarse = 0.00016
    densify_grad_threshold_fine_init = 0.00016
    densify_grad_threshold_after = 0.00016
    pruning_from_iter = 500
    pruning_interval = 100
    opacity_threshold_coarse = 0.005
    opacity_threshold_fine_init = 0.005
    opacity_threshold_fine_after = 0.005
    add_point = False

    # Underwater physics optimization
    uw_warmup_iters = 3000
    # With physical-depth water and clamped clean radiance, the underwater
    # terms are real constraints rather than easy shortcuts. Ramp them past the
    # densification window so geometry can settle before full medium pressure.
    uw_aux_ramp_iters = 9000
    uw_aux_peak_iter = 12000
    uw_aux_decay_end_iter = 20000
    uw_aux_decay_min = 0.20             # was 0.35 — lower floor after decay
    uw_aux_w_max = 0.30                 # hard cap: UW aux never exceeds 30% of photo weight
    uw_lr_sigma = 0.001
    # The medium is 18 well-bounded params; it must adapt FASTER than the
    # scene, not slower. At 5e-4 + clip 0.1 it moved ~0.04/4k iters while J
    # and geometry (millions of params) absorbed the prior mismatch instead
    # (orange J, blurred geometry — the v13 set5 collapse).
    uw_lr_planes = 0.005
    uw_tau_motion = 0.05
    # Medium pre-calibration at fine start (v14)
    uw_precalib_views = 24
    uw_precalib_iters = 400
    uw_sigma_thresh = 0.50
    # The old 0.5 one-shot inversion made the clean branch start warm/orange.
    # Keep it off by default; if used as an ablation, keep the blend gentle.
    uw_color_inversion = False
    uw_color_inversion_blend = 0.1
    uw_color_inversion_min_trans = 0.25

    # v12 calibration: reduced after observing n_transient=57k (30% false positives)
    # and uw_effective_w oscillating between 0 and 0.58 in v11 runs.
    uw_lambda_transient = 0.01          # was 0.03 — less pressure to flag Gaussians
    uw_lambda_transient_residual = 0.005 # was 0.02 — RGSSA was main false-positive driver
    uw_lambda_sparse = 0.0002
    uw_sigma_target_mean = 0.05         # was 0.08 — lower target, fewer false flags
    uw_sigma_lower_bound = 0.01         # was 0.03
    uw_sigma_upper_bound = 0.25         # was 0.20 — more headroom before clamping
    uw_sigma_binarize_weight = 0.02     # was 0.05 — softer binarization pressure
    # Physics losses halved from v11 since uw_aux_w now caps at 0.30
    uw_lambda_dark = 0.03               # v15: 0.05 -> 0.03 (DCP over-darkening fed yellow shift)
    uw_lambda_binf = 0.02               # was 0.05
    uw_lambda_backscatter_dark = 0.02   # was 0.05
    # New losses
    uw_lambda_physics_ordering = 0.05   # unchanged — prevents J collapsing to I
    uw_lambda_sigma_residual = 0.005    # was 0.02 — key false-positive fix
    uw_lambda_depth_weighted = 0.01
    uw_lambda_mono_depth = 0.01
    uw_lambda_depth_tv = 0.002
    uw_lambda_tone = 0.01
    uw_lambda_uncertainty = 0.005
    uw_lambda_exposure = 0.005
    # This is no longer a strict gray-world loss; it is a weak color-cast prior.
    # J-side hue anchor (anchor group in train.py). v14's 0.002 at cosmetic
    # weight let B_inf over-subtraction invert the color cast (yellow J).
    # 0.05: at 0.02 the dead-zone gradient was ~1e-3, an order of magnitude
    # below the photometric pull that inflates the medium.
    uw_lambda_gray = 0.05
    # Anchor B_inf to its open-water measurement (zero-opacity rays image the
    # water column, which saturates to B_inf — a direct observation). Strong:
    # it is a measurement of the exact physical quantity, not a prior.
    uw_lambda_medium_prior = 0.5
    # v16: scene-agnostic anti-shift prior — restored J must be statistically
    # independent of range. Catches depth-graded hue drift of any direction
    # (yellow/red/green/blue) on any water type without hue-specific margins.
    uw_lambda_depth_decorr = 0.05
    # v15: stop transients entering the scene representation
    uw_robust_photo = True
    uw_robust_floor = 0.2
    uw_lambda_transient_opacity = 0.01
    # v15: capacity
    uw_max_points = 700000
    uw_fine_opacity_prune = True
    uw_binf_percentile = 0.6
    uw_lambda_saturation = 0.005
    uw_lambda_spectral = 0.005

    # Underwater loss controls.
    uw_mono_depth_residual_clip = 2.0
    uw_mono_depth_inverse = True
    uw_depth_weight_gamma = 1.0
    uw_tone_mu = 10.0
    uw_uncertainty_sharpness = 6.0
    uw_exposure_upper = 0.92
    uw_saturation_thresh = 0.98
    uw_photo_guard_start = 0.05
    # v12: reduced from 0.25 → 0.15. The v11 value of 0.25 was too loose —
    # photo guard was hitting 0 oscillation when photo_l1 spiked above 0.25.
    # The guard now uses squared falloff (in train.py) which is smoother.
    uw_photo_guard_stop = 0.15
    uw_dark_threshold = 0.18
    uw_dark_fraction_threshold = 0.01
    uw_dark_min_weight = 0.20
    uw_binf_dark_thresh = 0.25
    uw_binf_min_depth = 0.25
    uw_anchor_min_pixels = 256
    uw_backscatter_dark_thresh = 0.25
    uw_backscatter_min_depth = 0.25

    # The underwater fine branch can temporarily make opacity/photometric
    # gradients noisy. Keep geometry alive while the medium parameters settle.
    uw_disable_fine_pruning = True
    uw_disable_fine_opacity_reset = True
    uw_min_gaussians = 20000
    uw_max_prune_fraction = 0.25


class PipelineParams:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


class ModelHiddenParams:
    net_width = 128   # v15: upstream 4DGS hypernerf size; 64 underfit dynamics (flicker)
    timebase_pe = 4
    defor_depth = 1
    posebase_pe = 10
    scale_rotation_pe = 2
    opacity_pe = 2
    timenet_width = 64
    timenet_output = 32
    bounds = 1.6

    plane_tv_weight = 0.0001
    time_smoothness_weight = 0.01
    l1_time_planes = 0.0001

    kplanes_config = {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 32,
        # set4 has 300 sampled frames, so temporal resolution 150 is half-rate.
        # This gives dynamic capacity without assigning one bin per image.
        "resolution": [64, 64, 64, 150],
    }
    multires = [1, 2, 4]   # v15: match upstream; [1,2] lacked fine spatial detail

    no_dx = False
    no_grid = False
    no_ds = False
    no_dr = False
    no_do = True
    no_dshs = True
    empty_voxel = False
    grid_pe = 0
    static_mlp = False
    apply_rotation = False
