"""
gaussian_renderer/renderer_uw.py — PhysSplit-4DGS

PER-PIXEL PHYSICS:
    1. Rasterize Gaussians normally → J (rendered clean colour) + depth map
    2. Query physics_planes at each pixel's normalised depth → β_D(z) per pixel
    3. Apply Akkaynak forward pixel-wise → I_degraded
    4. Return both J (for DCP loss, restoration) and I_degraded (vs GT for L_photo)
"""

import math
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                          GaussianRasterizer,
                                          underwater_image_formation)
from gaussian_renderer import render as _render_orig


def _get_raster_settings(viewpoint_camera, pc, pipe, bg_color,
                         scaling_modifier, cam_type=None):
    if cam_type == "PanopticSports" or isinstance(viewpoint_camera, dict):
        settings = viewpoint_camera['camera']
        return GaussianRasterizationSettings(
            image_height=settings.image_height,
            image_width=settings.image_width,
            tanfovx=settings.tanfovx,
            tanfovy=settings.tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=settings.viewmatrix,
            projmatrix=settings.projmatrix,
            sh_degree=pc.active_sh_degree,
            campos=settings.campos,
            prefiltered=False,
            debug=pipe.debug,
        )
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )


def _pixel_range_scale(viewpoint_camera, depth_z, cam_type=None):
    """
    Convert camera z-depth to optical path range.

    Underwater attenuation/backscatter depends on camera-object range along the
    viewing ray, not merely the projected z coordinate. For a pinhole ray
    (x*tan(fovx), y*tan(fovy), 1), range = z * ||ray||.
    """
    H, W = depth_z.shape
    device = depth_z.device
    dtype = depth_z.dtype

    if cam_type == "PanopticSports" or isinstance(viewpoint_camera, dict):
        settings = viewpoint_camera["camera"]
        tanfovx = float(settings.tanfovx)
        tanfovy = float(settings.tanfovy)
    else:
        tanfovx = math.tan(float(viewpoint_camera.FoVx) * 0.5)
        tanfovy = math.tan(float(viewpoint_camera.FoVy) * 0.5)

    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / max(H, 1) * 2.0 - 1.0
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / max(W, 1) * 2.0 - 1.0
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")
    ray_x = xs * tanfovx
    ray_y = ys * tanfovy
    return torch.sqrt(ray_x.square() + ray_y.square() + 1.0)


def render_uw(viewpoint_camera, pc, pipe, bg_color,
               scaling_modifier=1.0, override_color=None,
               stage="fine", cam_type=None,
               physics_planes=None,
               scene_bbox_min=None, scene_bbox_max=None,
               scene_depth_max=None,
               mono_depth_inverse=False):
    """
    PhysSplit-4DGS forward render — per-pixel physics.
    """

    # Sanitise Gaussian parameters
    with torch.no_grad():
        if not torch.isfinite(pc._xyz).all():
            pc._xyz.data.nan_to_num_(nan=0.0, posinf=100.0, neginf=-100.0)
        if not torch.isfinite(pc._scaling).all():
            pc._scaling.data.nan_to_num_(nan=-2.0, posinf=5.0, neginf=-5.0)
        if not torch.isfinite(pc._rotation).all():
            pc._rotation.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
        pc._xyz.data.clamp_(-100.0, 100.0)
        pc._scaling.data.clamp_(-5.0, 5.0)

    # If physics disabled, fall back to standard render
    if physics_planes is None:
        return dict(_render_orig(
            viewpoint_camera, pc, pipe, bg_color,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            stage=stage, cam_type=cam_type,
        ))

    # Run deformation (4DGS standard)
    means3D = pc.get_xyz
    scales = pc._scaling
    rotations = pc._rotation
    opacity = pc._opacity
    shs = pc.get_features

    if cam_type != "PanopticSports":
        time = torch.tensor(viewpoint_camera.time).to(
            means3D.device).repeat(means3D.shape[0], 1)
    else:
        time = torch.tensor(viewpoint_camera['time']).to(
            means3D.device).repeat(means3D.shape[0], 1)

    if "coarse" in stage:
        means3D_final = means3D
        scales_final = scales
        rotations_final = rotations
        opacity_final = opacity
        shs_final = shs
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
            pc._deformation(means3D, scales, rotations, opacity, shs, time)
    else:
        raise NotImplementedError(f"Unknown stage: {stage}")

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_act = pc.opacity_activation(opacity_final)

    # Rasterize J (clean colour) and depth
    screenspace_points = torch.zeros_like(
        means3D_final, dtype=means3D_final.dtype,
        requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    cov3D_precomp = None
    raster_scales = scales_final
    raster_rotations = rotations_final
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        raster_scales = None
        raster_rotations = None

    raster_settings = _get_raster_settings(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier, cam_type=cam_type)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 1. Render accumulated opacity (1 - T) by setting colors_precomp = ones and bg = zeros
    bg_black = torch.zeros(3, dtype=torch.float32, device="cuda")
    raster_settings_black = _get_raster_settings(
        viewpoint_camera, pc, pipe, bg_black, scaling_modifier, cam_type=cam_type)
    rasterizer_black = GaussianRasterizer(raster_settings=raster_settings_black)
    ones_color = torch.ones((means3D_final.shape[0], 3), dtype=torch.float32, device="cuda")
    accum_opacity_3d, _, _ = rasterizer_black(
        means3D=means3D_final,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=ones_color,
        opacities=opacity_act,
        scales=raster_scales,
        rotations=raster_rotations,
        cov3D_precomp=cov3D_precomp,
    )
    accum_opacity = accum_opacity_3d[0].clamp(0.0, 1.0) # (H, W)

    # 2. Render per-pixel mean σ (for RGSSA spatial residual sigma loss)
    sigma_render = None
    if hasattr(pc, "get_sigma"):
        sigma_vals = pc.get_sigma.clamp(0.0, 1.0)              # (N,) or (N,1)
        sigma_vals = sigma_vals.view(-1, 1)                    # (N, 1)
        sigma_colors = sigma_vals.expand(-1, 3)                 # (N, 3)
        sigma_raw_3d, _, _ = rasterizer_black(
            means3D=means3D_final,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=sigma_colors,
            opacities=opacity_act,
            scales=raster_scales,
            rotations=raster_rotations,
            cov3D_precomp=cov3D_precomp,
        )
        # Alpha-normalise to get mean sigma per pixel; shape (1, H, W)
        sigma_render = (sigma_raw_3d[0:1] / accum_opacity.unsqueeze(0).clamp_min(1e-5))
        sigma_render = sigma_render.clamp(0.0, 1.0)

    # 3. Render actual clean color J and raw depth
    rendered_J, radii, depth_raw_3d = rasterizer(
        means3D=means3D_final,
        means2D=screenspace_points,
        shs=shs_final,
        colors_precomp=None,
        opacities=opacity_act,
        scales=raster_scales,
        rotations=raster_rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # rendered_J: (3, H, W)

    # Process depth: alpha-normalized expected z, then z->range. The underwater
    # image formation equation uses camera-object range/optical path length.
    depth_raw = depth_raw_3d.squeeze(0) if depth_raw_3d.dim() == 3 else depth_raw_3d
    depth_expected_z = depth_raw / accum_opacity.clamp_min(1e-5)
    ray_scale = _pixel_range_scale(viewpoint_camera, depth_expected_z,
                                   cam_type=cam_type)
    depth_expected = depth_expected_z * ray_scale

    if scene_depth_max is None or scene_depth_max <= 0:
        # Avoid using corrupted zero depth pixels for scene max estimation
        valid_depths = depth_expected[accum_opacity > 0.05]
        if valid_depths.numel() > 0:
            scene_depth_max = valid_depths.max().item() + 1e-6
        else:
            scene_depth_max = depth_raw.max().item() + 1e-6
    scene_depth_max = float(scene_depth_max)

    # Use monocular depth map to fill in background pixels (where opacity is low)
    fg_mask = accum_opacity > 0.01
    mono_depth = None
    if cam_type != "PanopticSports" and not isinstance(viewpoint_camera, dict):
        mono_depth = getattr(viewpoint_camera, "depth", None)
    
    if mono_depth is not None:
        mono_depth = mono_depth.to(device=depth_raw.device, dtype=depth_raw.dtype)
        if mono_depth_inverse:
            mono_depth = 1.0 - mono_depth
        bg_depth = mono_depth.clamp(0.0, 1.0) * scene_depth_max
    else:
        bg_depth = torch.ones_like(depth_expected) * scene_depth_max

    depth_expected_final = torch.where(fg_mask, depth_expected, bg_depth)
    depth_norm = (depth_expected_final / scene_depth_max).clamp(0.0, 1.0)

    # Query physics_planes per pixel.
    # STOP-GRADIENT on depth: the physics composition reads depth as data.
    # If the photometric gradient flows through exp(-beta*d) into rasterized
    # depth, the optimizer uses per-pixel depth as a free color knob and
    # shreds geometry into floaters (densification then clones them).
    # Geometry must learn depth only via the J path and explicit depth losses.
    depth_norm_sg = depth_norm.detach()
    H, W = depth_norm.shape
    d_flat = depth_norm_sg.reshape(-1)
    phys = physics_planes.query(d_flat)
    beta_D_img = phys["beta_D"].reshape(H, W, 3)
    beta_B = phys["beta_B"]
    B_inf = phys["B_inf"]
    depth_scale = phys.get("depth_scale", torch.as_tensor(
        3.0, device=depth_norm.device, dtype=depth_norm.dtype))
    # SfM depth is not metric. Use normalized depth times a learned equivalent
    # optical path length rather than assuming scene units are metres.
    # (depth_scale still receives gradient through the product.)
    depth_physical = depth_norm_sg * depth_scale

    # The underwater image formation model is physical radiance math, so it
    # must not receive negative SH/raster overshoot values. Use a straight-
    # through clamp: forward values are valid [0, 1] radiance, while gradients
    # can still pull the Gaussian colors back into range.
    rendered_J_physical = rendered_J + (
        rendered_J.clamp(0.0, 1.0) - rendered_J).detach()

    # Gaussian Splashing style: rasterize clean J, then apply the forward
    # underwater image formation model in the rasterizer extension.
    I_degraded_chw, backscatter_chw = underwater_image_formation(
        clean_color=rendered_J_physical,
        depth=depth_physical,
        beta_D=beta_D_img,
        beta_B=beta_B,
        B_inf=B_inf,
    )

    # NaN safety
    def _check(name, x):
        if not torch.isfinite(x).all():
            print(f"[RENDER_UW] {name} NaN/Inf — fallback")
            return True
        return False

    bad = any([
        _check("depth_norm", depth_norm),
        _check("depth_physical", depth_physical),
        _check("I_degraded", I_degraded_chw),
        _check("rendered_J", rendered_J_physical),
    ])
    if bad:
        zero_backscatter = torch.zeros_like(rendered_J)
        return {
            "render": rendered_J_physical.clamp(0, 1),
            "render_degraded": rendered_J_physical.clamp(0, 1),
            "render_clean": rendered_J_physical.clamp(0, 1),
            "render_J": rendered_J_physical.clamp(0, 1),
            "rendered_J": rendered_J_physical.clamp(0, 1),
            "render_backscatter": zero_backscatter,
            "rendered_backscattering": zero_backscatter,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth_raw_3d,
            "depth_raw": depth_raw,
            "depth_expected": depth_expected_final,
            "depth_expected_z": depth_expected_z,
            "ray_scale": ray_scale,
            "depth_norm": depth_norm,
            "depth_physical": depth_physical,
            "depth_for_physics": depth_norm,
            "accum_opacity": accum_opacity,
            "beta_D_img": beta_D_img,
            "beta_B": beta_B,
            "B_inf": B_inf,
            "depth_scale": depth_scale,
            "sigma_render": sigma_render,
        }

    return {
        "render":            I_degraded_chw,
        "render_degraded":   I_degraded_chw,
        "render_clean":      rendered_J_physical.clamp(0, 1),
        "render_J":          rendered_J_physical.clamp(0, 1),
        "rendered_J":        rendered_J_physical.clamp(0, 1),
        "render_backscatter": backscatter_chw.clamp(0, 1),
        "rendered_backscattering": backscatter_chw.clamp(0, 1),
        "viewspace_points":  screenspace_points,
        "visibility_filter": radii > 0,
        "radii":             radii,
        "depth":             depth_raw_3d,
        "depth_raw":         depth_raw,
        "depth_expected":    depth_expected_final,
        "depth_expected_z":  depth_expected_z,
        "ray_scale":         ray_scale,
        "depth_norm":        depth_norm,
        "depth_physical":    depth_physical,
        "depth_for_physics":  depth_norm,
        "accum_opacity":     accum_opacity,
        "beta_D_img":        beta_D_img,
        "beta_B":            beta_B,
        "B_inf":             B_inf,
        "depth_scale":       depth_scale,
        "sigma_render":      sigma_render,
    }


@torch.no_grad()
def compute_motion_mask(pc, motion_thresh, n_times=8):
    """
    Per-Gaussian persistent-motion mask from the deformation field.

    Motion taxonomy at evaluation: sigma flags TRANSIENT content (what the
    deformation-conditioned forward model cannot explain), while this mask
    flags PERSISTENT DYNAMICS (content the deformation field tracks, e.g.
    swaying plants or resident fish). Suppressing both yields the static
    clean scene. motion_thresh is in world units of canonical displacement.
    """
    means = pc.get_xyz
    scales = pc._scaling
    rotations = pc._rotation
    opacity = pc._opacity
    shs = pc.get_features
    max_disp = torch.zeros(means.shape[0], device=means.device)
    for t in torch.linspace(0.0, 1.0, n_times):
        time = torch.full((means.shape[0], 1), float(t), device=means.device)
        m_t, _, _, _, _ = pc._deformation(means, scales, rotations,
                                          opacity, shs, time)
        max_disp = torch.maximum(max_disp, (m_t - means).norm(dim=-1))
    return max_disp > motion_thresh


def render_clean(viewpoint_camera, pc, pipe, bg_color,
                  physics_planes, sigma_thresh=0.5,
                  stage="fine", cam_type=None,
                  scene_bbox_min=None, scene_bbox_max=None,
                  scene_depth_max=None,
                  mono_depth_inverse=False,
                  motion_thresh=None):
    """
    Evaluation render with transient suppression; optionally also suppresses
    persistent-dynamic content (motion_thresh) for static scene extraction.
    """
    if not hasattr(pc, "get_sigma"):
        return render_uw(viewpoint_camera, pc, pipe, bg_color,
                          stage=stage, cam_type=cam_type,
                          physics_planes=physics_planes,
                          scene_depth_max=scene_depth_max,
                          mono_depth_inverse=mono_depth_inverse)

    sigma = pc.get_sigma.detach()
    orig_opa = pc._opacity.data.clone()

    with torch.no_grad():
        mask = sigma >= sigma_thresh
        if motion_thresh is not None:
            mask = mask | compute_motion_mask(pc, motion_thresh)
        if mask.any():
            pc._opacity.data[mask] = -10.0

    try:
        return render_uw(viewpoint_camera, pc, pipe, bg_color,
                          stage=stage, cam_type=cam_type,
                          physics_planes=physics_planes,
                          scene_depth_max=scene_depth_max,
                          mono_depth_inverse=mono_depth_inverse)
    finally:
        pc._opacity.data.copy_(orig_opa)
