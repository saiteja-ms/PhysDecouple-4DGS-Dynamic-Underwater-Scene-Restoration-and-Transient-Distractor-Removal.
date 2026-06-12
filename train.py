"""
train.py — PhysSplit-4DGS

KEY CHANGES vs previous version:
    1. RESTORED physics_planes=physics_planes in fine-stage call (was commented out)
    2. DELETED cross-frame SeaThru regression block
    3. DELETED physics_collapse_prevention loss term
    4. ADDED conditional dark channel loss on J
    5. ADDED dark-pixel B_∞ anchor (Sea-thru §4.3)
    6. NEW PhysicsPlanes signature: no depth_res/time_res params
    7. NEW UnderwaterGaussianModel: only _sigma per-Gaussian (no _beta_D, _beta_B)

LOSS SET (final):
    L_photo (L1 + SSIM)             - photometric primary
    + soft-gated L_dark             - DCP on J, no dead hard gate
    + top-k L_binf_anchor           - direct B_∞ estimate from dark/deep pixels
    + L_transient + budgeted L_sigma - σ machinery without all-zero collapse
"""

import numpy as np
import random
import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import (ModelParams, PipelineParams, OptimizationParams,
                        ModelHiddenParams)
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image
from time import time
import copy
import importlib.util

# PhysSplit-4DGS imports
from scene.gaussian_model_uw import UnderwaterGaussianModel
from scene.physics_planes import PhysicsPlanes
from gaussian_renderer.renderer_uw import render_uw
from utils.loss_uw import (uw_total_loss,
                            conditional_dark_channel_loss,
                            dark_pixel_binf_anchor,
                            dark_pixel_backscatter_loss,
                            depth_color_decorrelation_loss,
                            depth_weighted_l1_loss,
                            edge_aware_depth_tv_loss,
                            exposure_loss,
                            gray_world_loss,
                            physics_ordering_loss,
                            saturation_loss,
                            spatial_residual_sigma_loss,
                            spectral_ordering_loss,
                            tone_mapped_l1_loss,
                            uncertainty_weighted_l1_loss)
from utils.depth_utils import pearson_depth_loss
from utils.viser_viewer import LiveViewer

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    print("[WARN] TensorBoard not found — no scalars will be logged")


def measure_binf_open_water(scene, gaussians, pipe, background,
                             physics_planes, args, scene_depth_max,
                             n_views=24, opacity_thresh=0.05,
                             quantile=0.9, min_pixels=2048):
    """
    Measure B_inf from open water — physics, not a prior.

    In the Akkaynak formation model, a ray that intersects no scene geometry
    images the water column itself, whose radiance saturates to exactly B_inf
    with range. GT pixels where the coarse reconstruction has ~zero opacity
    are therefore direct observations of B_inf; the high quantile estimates
    the saturated asymptote robustly. Returns (3,) or None when the scene has
    no open water (then the dark-pixel Sea-thru estimator remains the anchor).
    """
    train_cams = scene.getTrainCameras()
    if len(train_cams) == 0:
        return None
    idxs = np.random.choice(len(train_cams),
                            size=min(n_views, len(train_cams)), replace=False)
    vals = []
    with torch.no_grad():
        for i in idxs:
            cam = train_cams[int(i)]
            pkg = render_uw(
                cam, gaussians, pipe, background, stage="fine",
                cam_type=scene.dataset_type, physics_planes=physics_planes,
                scene_depth_max=scene_depth_max,
                mono_depth_inverse=getattr(args, "uw_mono_depth_inverse", False))
            acc = pkg.get("accum_opacity")
            if acc is None:
                continue
            m = acc < opacity_thresh
            if m.any():
                if scene.dataset_type != "PanopticSports":
                    gt = cam.original_image.cuda()[:3]
                else:
                    gt = cam['image'].cuda()[:3]
                vals.append(gt[:, m].permute(1, 0))
    if not vals:
        return None
    v = torch.cat(vals, 0)
    if v.shape[0] < min_pixels:
        return None
    return torch.quantile(v.float(), quantile, dim=0).clamp(0.02, 0.98)


def precalibrate_medium(scene, gaussians, pipe, background, physics_planes,
                         args, scene_depth_max, n_views=24, iters=400):
    """Fit the 18 medium params to (J_coarse, GT) pairs before fine training.

    At fine start the coarse model has J ~= GT but the medium sits at Jerlov
    priors, so the composition I(J, priors) is far from GT. The medium is 18
    bounded params; the scene is millions of fast ones. Without this step the
    scene absorbs the prior mismatch (orange J, blurred geometry) long before
    the medium can adapt. Renders n_views once, then optimizes the medium
    alone on cached (J, depth, GT) triples — a few seconds of compute.
    """
    train_cams = scene.getTrainCameras()
    if len(train_cams) == 0 or iters <= 0:
        return
    idxs = np.random.choice(len(train_cams),
                            size=min(n_views, len(train_cams)), replace=False)
    cached = []
    with torch.no_grad():
        for i in idxs:
            cam = train_cams[int(i)]
            pkg = render_uw(
                cam, gaussians, pipe, background, stage="fine",
                cam_type=scene.dataset_type, physics_planes=physics_planes,
                scene_depth_max=scene_depth_max,
                mono_depth_inverse=getattr(args, "uw_mono_depth_inverse", False))
            if scene.dataset_type != "PanopticSports":
                gt = cam.original_image.cuda()
            else:
                gt = cam['image'].cuda()
            cached.append((pkg["render_J"].detach(),
                           pkg["depth_norm"].detach(), gt[:3]))

    # When B_inf was measured from open water it is data, not a fit variable.
    fit_params = [p for n, p in physics_planes.named_parameters()
                  if not (hasattr(physics_planes, "_measured_B_inf")
                          and n == "raw_Binf")]
    opt_p = torch.optim.Adam(fit_params, lr=1e-2)
    last = None
    for it in range(iters):
        J, dnorm, gt = cached[it % len(cached)]
        H, W = dnorm.shape
        phys = physics_planes.query(dnorm.reshape(-1))
        beta_D = phys["beta_D"].reshape(H, W, 3)
        d_phys = (dnorm * phys["depth_scale"]).unsqueeze(-1)
        T_D = torch.exp(-beta_D * d_phys)
        T_B = torch.exp(-phys["beta_B"] * d_phys)
        J_hwc = J.permute(1, 2, 0)
        I_hwc = J_hwc * T_D + phys["B_inf"] * (1.0 - T_B)
        I_chw = I_hwc.permute(2, 0, 1)
        loss = (I_chw - gt).abs().mean()
        loss = loss + 0.05 * dark_pixel_binf_anchor(
            I_observed=gt, J_rendered=J, depth=dnorm,
            B_inf_predicted=phys["B_inf"],
            percentile=getattr(args, "uw_binf_percentile", 0.6))
        loss = loss + 0.05 * physics_ordering_loss(J, I_chw.clamp(0, 1))
        opt_p.zero_grad()
        loss.backward()
        opt_p.step()
        last = loss.item()
    with torch.no_grad():
        p = physics_planes.query(torch.tensor([0.5], device="cuda"))
        print(f"[PRECALIB] {iters} steps, final loss {last:.4f} | "
              f"B_inf={p['B_inf'].cpu().numpy().round(3)} "
              f"beta_D(z=0.5)={p['beta_D'][0].detach().cpu().numpy().round(3)} "
              f"depth_scale={float(p['depth_scale']):.2f}")
        # The calibration is a per-scene measurement of physically constant
        # medium parameters. Store it as the center of a trust region: during
        # training the photometric J/B ambiguity otherwise re-inflates the
        # medium (B_inf/depth_scale up -> blue over-subtraction -> warm J).
        physics_planes._calib_B_inf = p["B_inf"].detach().clone()
        physics_planes._calib_beta_B = p["beta_B"].detach().clone()
        physics_planes._calib_depth_scale = p["depth_scale"].detach().clone()


def scene_reconstruction(dataset, opt, hyper, pipe,
                          testing_iterations, saving_iterations,
                          checkpoint_iterations, checkpoint, debug_from,
                          gaussians, scene, stage, tb_writer,
                          train_iter, timer,
                          physics_planes=None,
                          scene_bbox_min=None,
                          scene_bbox_max=None,
                          scene_depth_max=None,
                          args=None,
                          live_viewer=None):

    first_iter = 0
    gaussians.training_setup(opt)

    if checkpoint:
        ckpt_name = os.path.basename(checkpoint)
        if stage == "coarse" and "fine" in ckpt_name:
            print("Start from fine stage, skipping coarse.")
            return
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

        if physics_planes is not None:
            phys_ckpt = checkpoint.replace("chkpnt_", "physics_planes_")
            if os.path.exists(phys_ckpt):
                print(f"Restoring physics planes from {phys_ckpt}")
                try:
                    physics_planes.load_state_dict(
                        torch.load(phys_ckpt, map_location="cuda"))
                except Exception as e:
                    print(f"[WARN] couldn't load old physics_planes: {e}")
                    print(f"[WARN] starting from Jerlov priors")
            else:
                print("[WARN] physics planes checkpoint not found.")

    if (physics_planes is not None
            and stage == "fine"
            and not getattr(gaussians, "_uw_groups_added", False)):
        gaussians.min_points_after_prune = getattr(args, "uw_min_gaussians", 20000)
        gaussians.max_prune_fraction = getattr(args, "uw_max_prune_fraction", 0.25)
        lr_sigma = getattr(args, "uw_lr_sigma", 1e-3)
        lr_planes = getattr(args, "uw_lr_planes", 5e-4)
        for g in gaussians.training_setup_uw(lr_sigma=lr_sigma):
            gaussians.optimizer.add_param_group(g)
        gaussians.optimizer.add_param_group({
            "params": list(physics_planes.parameters()),
            "lr": lr_planes,
            "name": "physics_planes",
        })
        gaussians._uw_groups_added = True

        # Calibrate the medium to the coarse reconstruction before any fine
        # photometric step (skipped when resuming from a fine checkpoint).
        if first_iter == 0:
            bg_pre = torch.tensor(
                [1, 1, 1] if dataset.white_background else [0, 0, 0],
                dtype=torch.float32, device="cuda")
            # 1) Measure the veiling light from open-water rays (physics:
            #    zero-opacity rays image the water column, which saturates to
            #    B_inf). A direct measurement when the scene shows open water.
            B_meas = measure_binf_open_water(
                scene, gaussians, pipe, bg_pre, physics_planes, args,
                scene_depth_max)
            if B_meas is not None:
                physics_planes.set_Binf(B_meas)
                physics_planes._measured_B_inf = B_meas.detach().clone()
                print(f"[B_INF] measured from open water: "
                      f"{B_meas.cpu().numpy().round(3)}")
            else:
                print("[B_INF] no open-water pixels — using dark-pixel "
                      "estimator (Sea-thru) as anchor")
            # 2) Fit the remaining medium params against (J_coarse, GT).
            precalibrate_medium(
                scene, gaussians, pipe, bg_pre, physics_planes, args,
                scene_depth_max,
                n_views=int(getattr(args, "uw_precalib_views", 24)),
                iters=int(getattr(args, "uw_precalib_iters", 400)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter),
                         desc=f"Training [{stage}]", dynamic_ncols=True)
    first_iter += 1

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("Data loading done")

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack, batch_size=batch_size,
                sampler=sampler, num_workers=16, collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack, batch_size=batch_size,
                shuffle=True, num_workers=16, collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    count = 0
    render_pkg_keep = None
    I_observed_chw = None
    for iteration in range(first_iter, final_iter + 1):

        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (custom_cam, do_training,
                 pipe.convert_SHs_python,
                 pipe.compute_cov3D_python,
                 keep_alive, scaling_modifer) = network_gui.receive()
                if custom_cam is not None:
                    count += 1
                    viewpoint_index = count % len(video_cams)
                    if (count // len(video_cams)) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render(
                        custom_cam, gaussians, pipe, background,
                        scaling_modifer, stage=stage,
                        cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, 0, 1) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (iteration < int(opt.iterations)
                                     or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack, batch_size=opt.batch_size,
                        shuffle=True, num_workers=32, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(loader)
        else:
            idx = 0
            viewpoint_cams = []
            while idx < batch_size:
                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue

        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        gt_images = []
        radii_list = []
        vis_list = []
        vp_list = []
        render_pkg_keep = None
        mono_depth_keep = None
        bad_render = False

        for viewpoint_cam in viewpoint_cams:

            if physics_planes is not None and stage == "fine":
                render_pkg = render_uw(
                    viewpoint_cam, gaussians, pipe, background,
                    stage=stage, cam_type=scene.dataset_type,
                    physics_planes=physics_planes,
                    scene_bbox_min=scene_bbox_min,
                    scene_bbox_max=scene_bbox_max,
                    scene_depth_max=scene_depth_max,
                    mono_depth_inverse=getattr(args, "uw_mono_depth_inverse", False),
                )
                render_pkg_keep = render_pkg
                mono_depth_keep = getattr(viewpoint_cam, "depth", None)
            else:
                render_pkg = render(
                    viewpoint_cam, gaussians, pipe, background,
                    stage=stage, cam_type=scene.dataset_type)
                render_pkg_keep = render_pkg

            image = render_pkg["render"]
            if (physics_planes is not None and stage == "fine"
                    and args is not None and "render_J" in render_pkg):
                # Ease the coarse->fine handoff: blend the physics composition
                # in gradually. At fine iter 0 the coarse model has J ~= GT;
                # switching the photometric target to the fully-degraded I in
                # one step doubles the error and slams geometry with gradients
                # before the medium params have adapted.
                _blend_iters = max(1, int(getattr(args, "uw_warmup_iters", 3000)))
                _alpha_phys = min(1.0, iteration / float(_blend_iters))
                if _alpha_phys < 1.0:
                    image = (_alpha_phys * image
                             + (1.0 - _alpha_phys) * render_pkg["render_J"])
            viewspace_pt = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            if not torch.isfinite(image).all():
                print(f"[ITER {iteration}] Bad render — NaN, skipping")
                bad_render = True
                break

            images.append(image.unsqueeze(0))

            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam["image"].cuda()

            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            vis_list.append(visibility_filter.unsqueeze(0))
            vp_list.append(viewspace_pt)

        if bad_render:
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(vis_list, 0).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)

        # Primary photometric loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        # Robust (sigma-gated) photometric: downweight pixels the transient
        # head flags so distractors are never fitted into the scene as
        # view-overfit SH blobs (they then explode in color from novel views
        # and pop in/out across time). The floor keeps some gradient alive so
        # sigma false-positives cannot carve permanent holes; sigma itself
        # keeps learning from the unweighted residual via RGSSA.
        loss_l1 = Ll1
        if (physics_planes is not None and stage == "fine"
                and args is not None
                and getattr(args, "uw_robust_photo", True)
                and render_pkg_keep is not None
                and iteration > getattr(args, "uw_warmup_iters", 3000)):
            sig = render_pkg_keep.get("sigma_render")
            if sig is not None and len(images) == 1:
                w = (1.0 - sig.detach().clamp(0.0, 1.0)).clamp_min(
                    float(getattr(args, "uw_robust_floor", 0.2)))
                resid = (image_tensor[-1] - gt_image_tensor[-1, :3]).abs()
                loss_l1 = (w * resid).mean()
        loss = loss_l1

        if opt.lambda_dssim != 0:
            loss += opt.lambda_dssim * (1.0 - ssim(
                image_tensor, gt_image_tensor[:, :3, :, :]))

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight,
                hyper.l1_time_planes,
                hyper.plane_tv_weight)
            loss += tv_loss

        # UW losses (fine). Anchors run from iter 0 — the medium needs its
        # data-driven constraints exactly while the composition settles, not
        # after warmup. Cosmetic terms still ramp in via uw_aux_w (0 until
        # uw_warmup_iters by construction of the cosine ramp).
        uw_components = {}
        if (physics_planes is not None
                and stage == "fine"
                and render_pkg_keep is not None
                and args is not None):

            warmup_iter = getattr(args, "uw_warmup_iters", 3000)
            aux_ramp_iters = max(1, getattr(args, "uw_aux_ramp_iters", 5000))
            aux_peak_iter = getattr(
                args, "uw_aux_peak_iter", warmup_iter + aux_ramp_iters)
            aux_peak_iter = max(warmup_iter + 1, int(aux_peak_iter))
            aux_decay_end = max(
                aux_peak_iter + 1,
                int(getattr(args, "uw_aux_decay_end_iter", train_iter)))
            aux_min_w = float(getattr(args, "uw_aux_decay_min", 0.35))
            aux_min_w = min(1.0, max(0.0, aux_min_w))
            if iteration <= aux_peak_iter:
                ramp_t = min(1.0, max(0.0, (iteration - warmup_iter)
                                      / float(aux_peak_iter - warmup_iter)))
                uw_aux_w = 0.5 - 0.5 * np.cos(np.pi * ramp_t)
            else:
                decay_t = min(1.0, max(0.0, (iteration - aux_peak_iter)
                                       / float(aux_decay_end - aux_peak_iter)))
                decay = 0.5 + 0.5 * np.cos(np.pi * decay_t)
                uw_aux_w = aux_min_w + (1.0 - aux_min_w) * decay
            # Hard cap: never let UW auxiliary weight exceed 0.3 of photo weight.
            # Without this, uw_aux_w ramps to 1.0 by iter 12k and the combined
            # UW loss dominates photometric reconstruction, destroying J quality.
            uw_aux_w_max = float(getattr(args, "uw_aux_w_max", 0.30))
            uw_aux_w = min(uw_aux_w, uw_aux_w_max)

            photo_guard_start = getattr(args, "uw_photo_guard_start", 0.08)
            photo_guard_stop = getattr(args, "uw_photo_guard_stop", 0.18)
            photo_l1 = float(Ll1.detach().item())
            if photo_guard_stop <= photo_guard_start:
                uw_photo_w = 1.0
            elif photo_l1 <= photo_guard_start:
                uw_photo_w = 1.0
            elif photo_l1 >= photo_guard_stop:
                uw_photo_w = 0.0
            else:
                # Smooth squared falloff instead of linear — avoids hard on/off
                # oscillation when photo_l1 hovers near photo_guard_stop.
                t = ((photo_guard_stop - photo_l1)
                     / (photo_guard_stop - photo_guard_start))
                uw_photo_w = t * t
            uw_effective_w = uw_aux_w * uw_photo_w

            # Anchor weight for the physics-grounding losses (binf, dark,
            # backscatter-dark, ordering). NOT gated by the photo guard: high
            # photometric error is when the medium params drift fastest, i.e.
            # exactly when anchoring is needed. Gating them created a feedback
            # loop (high l1 -> anchors off -> B_inf/beta drift -> higher l1).
            uw_anchor_w = min(1.0, iteration / 500.0)

            N = gaussians.get_xyz.shape[0]
            if not hasattr(gaussians, "_deform_ema"):
                gaussians._deform_ema = None

            if (hasattr(gaussians, "_deformation_accum")
                    and gaussians._deformation_accum is not None
                    and gaussians._deformation_accum.shape[0] == N):
                current_norm = gaussians._deformation_accum.norm(dim=-1).detach()
                current_norm = torch.nan_to_num(current_norm, nan=0.0, posinf=0.0)
                if (gaussians._deform_ema is None
                        or gaussians._deform_ema.shape[0] != N):
                    gaussians._deform_ema = current_norm.clone()
                else:
                    gaussians._deform_ema = (0.99 * gaussians._deform_ema
                                              + 0.01 * current_norm)
                deform_norm = gaussians._deform_ema.clamp(0., 10.)
            else:
                deform_norm = torch.zeros(N, device="cuda")

            # σ machinery
            I_degraded = render_pkg_keep["render"]
            uw_loss, uw_components = uw_total_loss(
                sigma_hat=gaussians.get_sigma,
                deform_norm=deform_norm,
                opacity=gaussians.get_opacity.squeeze(-1),
                I_observed=gt_image_tensor[-1, :3].permute(1, 2, 0),
                I_predicted=I_degraded.permute(1, 2, 0),
                iteration=iteration,
                args=args,
            )
            loss = loss + uw_effective_w * uw_loss

            # Forward underwater restoration losses on the same camera package.
            J_rendered = render_pkg_keep["render_J"]
            I_observed_chw = gt_image_tensor[-1, :3]
            depth_norm = render_pkg_keep.get("depth_for_physics",
                                             render_pkg_keep.get("depth_norm"))

            l_dark = conditional_dark_channel_loss(
                J_rendered,
                dark_threshold=getattr(args, "uw_dark_threshold", 0.18),
                fraction_threshold=getattr(
                    args, "uw_dark_fraction_threshold", 0.01),
                min_weight=getattr(args, "uw_dark_min_weight", 0.20),
                I_observed=I_observed_chw)
            loss = loss + uw_anchor_w * getattr(args, "uw_lambda_dark", 0.05) * l_dark

            l_depth_weighted = depth_weighted_l1_loss(
                I_degraded, I_observed_chw, depth_norm,
                gamma=getattr(args, "uw_depth_weight_gamma", 1.0))
            loss = loss + uw_effective_w * getattr(args, "uw_lambda_depth_weighted", 0.01) * l_depth_weighted

            if mono_depth_keep is not None:
                mono_depth_target = mono_depth_keep.to(
                    device=depth_norm.device, dtype=depth_norm.dtype)
                if getattr(args, "uw_mono_depth_inverse", False):
                    mono_depth_target = 1.0 - mono_depth_target
                l_mono_depth = pearson_depth_loss(depth_norm, mono_depth_target)
            else:
                l_mono_depth = torch.zeros((), device=I_degraded.device)
            # Depth supervision is geometry anchoring — anchor group, not the
            # photo-guard-gated cosmetic group (which is ~0 when l1 is high).
            loss = loss + uw_anchor_w * getattr(args, "uw_lambda_mono_depth", 0.0005) * l_mono_depth

            l_depth_tv = edge_aware_depth_tv_loss(depth_norm, I_observed_chw)
            loss = loss + uw_anchor_w * getattr(args, "uw_lambda_depth_tv", 0.002) * l_depth_tv

            l_tone = tone_mapped_l1_loss(
                I_degraded, I_observed_chw,
                mu=getattr(args, "uw_tone_mu", 10.0))
            loss = loss + uw_effective_w * getattr(args, "uw_lambda_tone", 0.01) * l_tone

            l_uncertainty = uncertainty_weighted_l1_loss(
                I_degraded, I_observed_chw, depth_norm,
                sharpness=getattr(args, "uw_uncertainty_sharpness", 6.0))
            loss = loss + uw_effective_w * getattr(args, "uw_lambda_uncertainty", 0.005) * l_uncertainty

            l_exposure = exposure_loss(
                J_rendered, upper=getattr(args, "uw_exposure_upper", 0.92))
            loss = loss + uw_effective_w * getattr(args, "uw_lambda_exposure", 0.005) * l_exposure

            # Gray-world is the J-side hue anchor. It must carry real weight:
            # with it at cosmetic strength, B_inf over-subtraction inverts the
            # color cast (v14 set5: GT B-R +0.05 -> J B-R -0.27, yellow shift).
            l_gray = gray_world_loss(J_rendered)
            loss = loss + uw_anchor_w * getattr(args, "uw_lambda_gray", 0.02) * l_gray

            # Depth-color decorrelation: restored J must be statistically
            # independent of range (the medium is the only range-dependent
            # term in the formation model). Catches depth-graded hue drift of
            # ANY direction — global-mean priors are blind to it (far-field
            # maroon on deep SeaThru-NeRF scenes evaded both gray-world and
            # the medium trust region) — and constrains exactly the far-field
            # pixels where T_D -> 0 kills the photometric gradient on J.
            _acc = render_pkg_keep.get("accum_opacity")
            _fg = (_acc > 0.5) if _acc is not None else None
            l_depth_decorr = depth_color_decorrelation_loss(
                J_rendered, depth_norm, fg_mask=_fg)
            loss = loss + uw_anchor_w * getattr(
                args, "uw_lambda_depth_decorr", 0.05) * l_depth_decorr

            l_saturation = saturation_loss(
                J_rendered, threshold=getattr(args, "uw_saturation_thresh", 0.98))
            loss = loss + uw_effective_w * getattr(args, "uw_lambda_saturation", 0.005) * l_saturation

            l_spectral = spectral_ordering_loss(
                render_pkg_keep.get("beta_D_img"),
                render_pkg_keep.get("beta_B"),
                render_pkg_keep.get("B_inf"))
            loss = loss + uw_effective_w * getattr(args, "uw_lambda_spectral", 0.005) * l_spectral

            l_bs_dark = dark_pixel_backscatter_loss(
                I_observed=I_observed_chw,
                J_rendered=J_rendered,
                backscatter_pred=render_pkg_keep["render_backscatter"],
                depth_norm=depth_norm,
                dark_threshold=getattr(args, "uw_backscatter_dark_thresh", 0.25),
                min_depth=getattr(args, "uw_backscatter_min_depth", 0.25),
                min_pixels=getattr(args, "uw_anchor_min_pixels", 256))
            loss = loss + uw_anchor_w * getattr(args, "uw_lambda_backscatter_dark", 0.01) * l_bs_dark

            # Dark-pixel B_∞ estimator (Sea-thru §4.3) — fallback only. When
            # open water is visible, B_inf is directly measured and anchored
            # via l_medium_prior; the indirect estimator would fight it.
            depth_for_anchor = render_pkg_keep.get(
                "depth_for_physics", render_pkg_keep.get("depth_norm"))
            if hasattr(physics_planes, "_measured_B_inf"):
                l_binf = torch.zeros((), device=I_degraded.device)
            else:
                l_binf = dark_pixel_binf_anchor(
                    I_observed=I_observed_chw,
                    J_rendered=J_rendered,
                    depth=depth_for_anchor,
                    B_inf_predicted=render_pkg_keep["B_inf"],
                    dark_threshold=getattr(args, "uw_binf_dark_thresh", 0.25),
                    min_depth=getattr(args, "uw_binf_min_depth", 0.25),
                    min_pixels=getattr(args, "uw_anchor_min_pixels", 256),
                    # 0.9 overestimated B_inf in bright shallow scenes (no
                    # truly dark pixels) -> over-subtraction -> yellow J.
                    percentile=getattr(args, "uw_binf_percentile", 0.6),
                )
            loss = loss + uw_anchor_w * getattr(args, "uw_lambda_binf", 0.01) * l_binf

            # Physics ordering loss: water must attenuate red and add blue.
            # Prevents the degenerate J ≈ I_GT escape where the physics model
            # shrinks to near-identity and the Gaussians absorb the water cast.
            l_physics_ordering = physics_ordering_loss(
                J_rendered, I_degraded, depth_norm=depth_norm)
            loss = loss + uw_anchor_w * getattr(
                args, "uw_lambda_physics_ordering", 0.05) * l_physics_ordering

            # Transient opacity decay: flagged Gaussians fade out so the
            # static background behind them keeps receiving photometric
            # supervision. Paired with the robust photometric downweighting,
            # transients never solidify into the representation, so masked
            # renders have no disocclusion holes. sigma is detached — this
            # must not push sigma down to dodge the penalty.
            l_transient_opacity = torch.zeros((), device=I_degraded.device)
            if (hasattr(gaussians, "get_sigma")
                    and iteration > getattr(args, "uw_warmup_iters", 3000)):
                _flag = gaussians.get_sigma.detach() > getattr(
                    args, "uw_sigma_thresh", 0.5)
                if _flag.any():
                    l_transient_opacity = gaussians.get_opacity.squeeze(-1)[
                        _flag].mean()
            loss = loss + uw_anchor_w * getattr(
                args, "uw_lambda_transient_opacity", 0.01) * l_transient_opacity

            # Veiling-light anchor. When the scene shows open water, B_inf is
            # a direct measurement (zero-opacity rays image the water column,
            # which saturates to B_inf) — anchor to it strongly. Worst-case
            # backscatter subtraction is then bounded by the TRUE veil, so
            # over-subtraction hue shifts are excluded by construction; no
            # growth caps or table bounds needed. beta and depth_scale stay
            # free inside their physical (Jerlov/identifiability) bounds.
            l_medium_prior = torch.zeros((), device=I_degraded.device)
            if hasattr(physics_planes, "_measured_B_inf"):
                # Direct measurement: strong anchor.
                l_medium_prior = (physics_planes.get_Binf()
                                  - physics_planes._measured_B_inf).abs().mean()
                _w_prior = getattr(args, "uw_lambda_medium_prior", 0.5)
            elif hasattr(physics_planes, "_calib_B_inf"):
                # No open water visible: anchor the veil COLOR to its
                # calibration estimate (prevents the slow B_inf drift that
                # produced the v14 yellow shift) while beta_B/depth_scale
                # stay free inside physical bounds so the veil STRENGTH can
                # still transfer from J into the medium.
                l_medium_prior = (physics_planes.get_Binf()
                                  - physics_planes._calib_B_inf).abs().mean()
                _w_prior = getattr(args, "uw_lambda_medium_prior_calib", 0.15)
            else:
                _w_prior = 0.0
            loss = loss + uw_anchor_w * _w_prior * l_medium_prior

            # RGSSA — Residual-Guided Spatial Sigma Assignment (novel).
            # Align the per-pixel rendered sigma to the per-pixel photometric
            # residual so transients are detected from the forward model's own
            # error map, complementing the deformation-based transient signal.
            sigma_render = render_pkg_keep.get("sigma_render", None)
            if sigma_render is not None:
                l_sigma_residual = spatial_residual_sigma_loss(
                    sigma_render, I_degraded, I_observed_chw)
            else:
                l_sigma_residual = torch.zeros((), device=I_degraded.device)
            loss = loss + uw_effective_w * getattr(
                args, "uw_lambda_sigma_residual", 0.02) * l_sigma_residual

            uw_components["uw_aux_w"] = torch.tensor(
                uw_aux_w, device=I_degraded.device)
            uw_components["uw_photo_w"] = torch.tensor(
                uw_photo_w, device=I_degraded.device)
            uw_components["uw_effective_w"] = torch.tensor(
                uw_effective_w, device=I_degraded.device)
            uw_components["uw_anchor_w"] = torch.tensor(
                uw_anchor_w, device=I_degraded.device)
            uw_components["l_dark"] = l_dark
            uw_components["l_binf"] = l_binf
            uw_components["l_depth_weighted"] = l_depth_weighted
            uw_components["l_mono_depth"] = l_mono_depth
            uw_components["l_depth_tv"] = l_depth_tv
            uw_components["l_tone"] = l_tone
            uw_components["l_uncertainty"] = l_uncertainty
            uw_components["l_exposure"] = l_exposure
            uw_components["l_gray"] = l_gray
            uw_components["l_depth_decorr"] = l_depth_decorr
            uw_components["l_saturation"] = l_saturation
            uw_components["l_spectral"] = l_spectral
            uw_components["l_backscatter_dark"] = l_bs_dark
            uw_components["l_physics_ordering"] = l_physics_ordering
            uw_components["l_transient_opacity"] = l_transient_opacity
            uw_components["l_medium_prior"] = l_medium_prior
            uw_components["l_sigma_residual"] = l_sigma_residual

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ITER {iteration}] NaN/Inf loss — skipping step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        for group in gaussians.optimizer.param_groups:
            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue
            name = group.get("name", "")
            if name == "physics_planes":
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)
            elif name == "sigma":
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)
        if hasattr(gaussians, "_deformation"):
            torch.nn.utils.clip_grad_norm_(
                gaussians._deformation.parameters(), max_norm=1.0)

        viewspace_point_tensor_grad = torch.zeros_like(vp_list[0])
        for vp in vp_list:
            if vp.grad is not None:
                viewspace_point_tensor_grad += vp.grad
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_.item() + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                    "pts": f"{total_point}",
                })
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            timer.pause()
            report_render = render
            report_kwargs = {}
            if physics_planes is not None and stage == "fine":
                report_render = render_uw
                report_kwargs = dict(
                    physics_planes=physics_planes,
                    scene_bbox_min=scene_bbox_min,
                    scene_bbox_max=scene_bbox_max,
                    scene_depth_max=scene_depth_max,
                    mono_depth_inverse=getattr(args, "uw_mono_depth_inverse", False),
                )
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                              iter_start.elapsed_time(iter_end),
                              testing_iterations, scene, report_render,
                              [pipe, background], stage, scene.dataset_type,
                              report_kwargs)

            if (tb_writer and iteration % 100 == 0
                    and stage == "fine" and uw_components):
                try:
                    for k, v in uw_components.items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            tb_writer.add_scalar(f"uw/{k}", v.item(), iteration)
                    if hasattr(gaussians, "get_sigma"):
                        sig = gaussians.get_sigma.detach()
                        tb_writer.add_scalar("uw/sigma_mean",
                                              sig.mean().item(), iteration)
                        tb_writer.add_scalar("uw/sigma_max",
                                              sig.max().item(), iteration)
                        tb_writer.add_scalar("uw/n_transient",
                                              (sig > 0.5).sum().item(), iteration)
                    if physics_planes is not None:
                        with torch.no_grad():
                            bd_s = physics_planes.get_betaD_at_depth(
                                torch.tensor([0.0], device="cuda"))
                            bd_m = physics_planes.get_betaD_at_depth(
                                torch.tensor([0.5], device="cuda"))
                            bd_d = physics_planes.get_betaD_at_depth(
                                torch.tensor([1.0], device="cuda"))
                            bb = physics_planes.get_betaB()
                            binf = physics_planes.get_Binf()
                            for ci, cname in enumerate(["R", "G", "B"]):
                                tb_writer.add_scalar(
                                    f"uw/beta_D_shallow_{cname}",
                                    bd_s[0, ci].item(), iteration)
                                tb_writer.add_scalar(
                                    f"uw/beta_D_mid_{cname}",
                                    bd_m[0, ci].item(), iteration)
                                tb_writer.add_scalar(
                                    f"uw/beta_D_deep_{cname}",
                                    bd_d[0, ci].item(), iteration)
                                tb_writer.add_scalar(
                                    f"uw/beta_B_{cname}", bb[ci].item(), iteration)
                                tb_writer.add_scalar(
                                    f"uw/B_inf_{cname}", binf[ci].item(), iteration)
                    if (render_pkg_keep is not None and iteration % 200 == 0
                            and "render_degraded" in render_pkg_keep):
                        tb_writer.add_image(
                            "uw/1_degraded_I",
                            render_pkg_keep["render_degraded"].clamp(0, 1),
                            iteration)
                        tb_writer.add_image(
                            "uw/2_clean_J",
                            render_pkg_keep["render_J"].clamp(0, 1),
                            iteration)
                        # Backscatter map
                        if "render_backscatter" in render_pkg_keep:
                            tb_writer.add_image(
                                "uw/3_backscatter",
                                render_pkg_keep["render_backscatter"].clamp(0, 1),
                                iteration)
                        # Sigma (transient score) heatmap
                        if render_pkg_keep.get("sigma_render") is not None:
                            sig_img = render_pkg_keep["sigma_render"].clamp(0, 1)
                            if sig_img.dim() == 3 and sig_img.shape[0] == 1:
                                sig_img = sig_img.expand(3, -1, -1)
                            tb_writer.add_image("uw/4_sigma_transient", sig_img, iteration)
                        # Depth map (normalized for display)
                        if "depth_norm" in render_pkg_keep:
                            d = render_pkg_keep["depth_norm"]
                            if d.dim() == 2:
                                d = d.unsqueeze(0)
                            if d.dim() == 3 and d.shape[0] == 1:
                                d = d.expand(3, -1, -1)
                            tb_writer.add_image("uw/5_depth_norm", d.clamp(0, 1), iteration)
                        # Side-by-side: GT | I_degraded | clean_J
                        if I_observed_chw is not None:
                            try:
                                gt_img = I_observed_chw.clamp(0, 1)
                                deg_img = render_pkg_keep["render_degraded"].clamp(0, 1)
                                cln_img = render_pkg_keep["render_J"].clamp(0, 1)
                                # Resize all to same size if needed
                                H = min(gt_img.shape[1], 270)
                                W = min(gt_img.shape[2], 480)
                                import torch.nn.functional as _F
                                def _resize(x):
                                    return _F.interpolate(
                                        x.unsqueeze(0), size=(H, W),
                                        mode="bilinear", align_corners=False).squeeze(0)
                                grid = torch.cat([_resize(gt_img),
                                                  _resize(deg_img),
                                                  _resize(cln_img)], dim=2)
                                tb_writer.add_image("uw/0_compare_GT|I|J", grid, iteration)
                            except Exception:
                                pass
                    tb_writer.flush()
                except Exception as e:
                    print(f"[TB] logging failed: {e}")

            # Live viser viewer update
            if live_viewer is not None and render_pkg_keep is not None:
                _gt_vis = I_observed_chw
                if _gt_vis is None and gt_image_tensor is not None:
                    _gt_vis = gt_image_tensor[-1, :3]
                live_viewer.update(iteration, render_pkg_keep, _gt_vis)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, stage)
                if physics_planes is not None:
                    torch.save(
                        physics_planes.state_dict(),
                        os.path.join(scene.model_path,
                                      f"physics_planes_{stage}_{iteration}.pth"))

            if dataset.render_process:
                if ((iteration < 1000 and iteration % 10 == 9)
                        or (iteration < 3000 and iteration % 50 == 49)
                        or (iteration < 60000 and iteration % 100 == 99)):
                    render_training_image(
                        scene, gaussians,
                        [test_cams[iteration % len(test_cams)]],
                        render, pipe, background,
                        stage + "test", iteration,
                        timer.get_elapsed_time(), scene.dataset_type)
                    render_training_image(
                        scene, gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render, pipe, background,
                        stage + "train", iteration,
                        timer.get_elapsed_time(), scene.dataset_type)
            timer.start()

            if iteration < opt.densify_until_iter:
                is_uw_fine = physics_planes is not None and stage == "fine"
                disable_uw_prune = (
                    is_uw_fine
                    and args is not None
                    and getattr(args, "uw_disable_fine_pruning", True)
                )
                disable_uw_reset = (
                    is_uw_fine
                    and args is not None
                    and getattr(args, "uw_disable_fine_opacity_reset", True)
                )

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = (
                        opt.opacity_threshold_fine_init
                        - iteration * (opt.opacity_threshold_fine_init
                                        - opt.opacity_threshold_fine_after)
                        / opt.densify_until_iter)
                    densify_threshold = (
                        opt.densify_grad_threshold_fine_init
                        - iteration * (opt.densify_grad_threshold_fine_init
                                        - opt.densify_grad_threshold_after)
                        / opt.densify_until_iter)

                # v14 froze at the old 350k cap from coarse onward: fine-stage
                # densification never ran, so fine details revealed by the
                # fine residuals could not be added ("finer features not
                # learnt"). Cap is now configurable and high.
                max_points = int(getattr(args, "uw_max_points", 700_000)
                                 if args is not None else 700_000)

                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                        and gaussians.get_xyz.shape[0] < max_points):
                    size_threshold = (20 if iteration > opt.opacity_reset_interval
                                       else None)
                    gaussians.densify(densify_threshold, opacity_threshold,
                                       scene.cameras_extent, size_threshold,
                                       5, 5, scene.model_path, iteration, stage)

                if (not disable_uw_prune
                        and iteration > opt.pruning_from_iter
                        and iteration % opt.pruning_interval == 0):
                    size_threshold = (20 if iteration > opt.opacity_reset_interval
                                       else None)
                    gaussians.prune(densify_threshold, opacity_threshold,
                                     scene.cameras_extent, size_threshold)
                elif (disable_uw_prune
                        and getattr(args, "uw_fine_opacity_prune", True)
                        and iteration > opt.pruning_from_iter
                        and iteration % opt.pruning_interval == 0):
                    # Opacity-only recycling in UW fine: no size/screen prune
                    # (that churn caused the v12 blur), but dead transparent
                    # Gaussians must be reclaimed so densification has budget.
                    # The UW prune() enforces min-points and max-fraction caps.
                    gaussians.prune(densify_threshold, opacity_threshold,
                                     scene.cameras_extent, None)

                if (iteration % opt.densification_interval == 0
                        and gaussians.get_xyz.shape[0] < max_points
                        and opt.add_point):
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)

                # Emergency cap prune. Must respect uw_disable_fine_pruning:
                # firing every 100 iters above the cap churns converged
                # Gaussians (prune big/transparent -> densify resplits ->
                # repeat) and blurs the reconstruction.
                if (not disable_uw_prune
                        and iteration % opt.densification_interval == 0
                        and gaussians.get_xyz.shape[0] > max_points):
                    size_threshold = 20
                    gaussians.prune(densify_threshold * 0.5, opacity_threshold,
                                     scene.cameras_extent, size_threshold)

                if (not disable_uw_reset
                        and iteration % opt.opacity_reset_interval == 0):
                    print("reset opacity")
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    gaussians._xyz.data.nan_to_num_(
                        nan=0.0, posinf=100.0, neginf=-100.0)
                    gaussians._xyz.data.clamp_(-100.0, 100.0)
                    gaussians._scaling.data.nan_to_num_(
                        nan=-2.0, posinf=5.0, neginf=-5.0)
                    gaussians._scaling.data.clamp_(-5.0, 5.0)
                    gaussians._rotation.data.nan_to_num_(
                        nan=0.0, posinf=1.0, neginf=-1.0)
                    gaussians._opacity.data.nan_to_num_(
                        nan=-1.0, posinf=5.0, neginf=-5.0)
                    if hasattr(gaussians, "_sigma"):
                        gaussians._sigma.data.nan_to_num_(
                            nan=-2.0, posinf=5.0, neginf=-5.0)
                    if physics_planes is not None:
                        physics_planes.raw_betaD_coeffs.data.nan_to_num_(
                            nan=0.0, posinf=5.0, neginf=-5.0)
                        physics_planes.raw_betaB.data.nan_to_num_(
                            nan=0.0, posinf=5.0, neginf=-5.0)
                        physics_planes.raw_Binf.data.nan_to_num_(
                            nan=0.0, posinf=5.0, neginf=-5.0)
                        physics_planes.raw_depth_scale.data.nan_to_num_(
                            nan=0.0, posinf=5.0, neginf=-5.0)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                            os.path.join(scene.model_path,
                                          f"chkpnt_{stage}_{iteration}.pth"))
                if physics_planes is not None:
                    torch.save(
                        physics_planes.state_dict(),
                        os.path.join(scene.model_path,
                                      f"physics_planes_{stage}_{iteration}.pth"))


def training(dataset, hyper, opt, pipe,
              testing_iterations, saving_iterations,
              checkpoint_iterations, checkpoint, debug_from, expname, args):

    tb_writer = prepare_output_and_logger(expname)

    gaussians = UnderwaterGaussianModel(
        dataset.sh_degree, hyper,
        water_type=getattr(args, "water_type", "II"),
    )

    physics_planes = PhysicsPlanes(
        water_type=getattr(args, "water_type", "II"),
    ).cuda()

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    with torch.no_grad():
        pts = gaussians.get_xyz.detach()
        scene_bbox_min = pts.min(dim=0).values
        scene_bbox_max = pts.max(dim=0).values

    run_stage = getattr(args, "stage", "both")

    # Live web viewer — open http://<server-ip>:7007 in any browser
    _viewer_port = getattr(args, "viewer_port", 7007)
    _viewer_every = getattr(args, "viewer_update_every", 200)
    live_viewer = None
    if not getattr(args, "disable_viewer", False):
        live_viewer = LiveViewer(port=_viewer_port, update_every=_viewer_every)

    if run_stage in ("coarse", "both"):
        scene_reconstruction(
            dataset, opt, hyper, pipe,
            testing_iterations, saving_iterations,
            checkpoint_iterations, checkpoint, debug_from,
            gaussians, scene, "coarse", tb_writer,
            opt.coarse_iterations, timer,
            physics_planes=None,
            scene_bbox_min=None,
            scene_bbox_max=None,
            scene_depth_max=None,
            args=None,
        )

    if run_stage == "coarse":
        return

    # Estimate scene_depth_max after coarse converges
    with torch.no_grad():
        pts = gaussians.get_xyz.detach()
        scene_bbox_min = pts.min(dim=0).values
        scene_bbox_max = pts.max(dim=0).values
        sample_depths = []
        train_cams_local = scene.getTrainCameras()
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        sample_indices = np.random.choice(
            len(train_cams_local),
            size=min(8, len(train_cams_local)),
            replace=False)
        for idx in sample_indices:
            try:
                cam = train_cams_local[int(idx)]
                pkg = render_uw(
                    cam, gaussians, pipe, background,
                    stage="coarse", cam_type=scene.dataset_type,
                    physics_planes=physics_planes,
                    scene_bbox_min=scene_bbox_min,
                    scene_bbox_max=scene_bbox_max,
                    scene_depth_max=None,
                    mono_depth_inverse=getattr(args, "uw_mono_depth_inverse", False),
                )
                d = pkg.get("depth_expected", None)
                if d is not None:
                    opacity = pkg.get("accum_opacity", None)
                    if opacity is not None:
                        d = d[opacity > 0.05]
                    if d.numel() > 0:
                        sample_depths.append(
                            torch.quantile(d.float(), 0.95).item())
            except Exception:
                continue
        if sample_depths:
            sample_depths = sorted(sample_depths)
            scene_depth_max = sample_depths[
                int(0.95 * (len(sample_depths) - 1))]
            print(f"[UW] Fixed scene_depth_max set to {scene_depth_max:.3f} "
                  f"(95th percentile of {len(sample_depths)} sample frames)")
        else:
            scene_depth_max = (scene_bbox_max - scene_bbox_min).norm().item()
            print(f"[UW] WARN: depth sampling failed, using bbox diag "
                  f"= {scene_depth_max:.3f}")
        try:
            with open(os.path.join(args.model_path,
                                     "scene_depth_max.txt"), "w") as f:
                f.write(f"{scene_depth_max:.6f}\n")
            print(f"[UW] saved scene_depth_max to "
                  f"{args.model_path}/scene_depth_max.txt")
        except Exception as e:
            print(f"[UW] WARN: failed to save scene_depth_max: {e}")

    if run_stage == "both":
        torch.cuda.empty_cache()

    # Reset/correct colors of Gaussians using the inverse Jerlov prior before fine-tuning (Fix 3)
    if run_stage in ("both", "fine") and physics_planes is not None and getattr(args, "uw_color_inversion", False):
        print("[UW] Performing coarse-to-fine Gaussian color correction (inverting physical prior)...")
        from utils.sh_utils import RGB2SH, SH2RGB
        with torch.no_grad():
            invert_blend = float(getattr(args, "uw_color_inversion_blend", 0.1))
            invert_blend = min(1.0, max(0.0, invert_blend))
            min_trans = float(getattr(args, "uw_color_inversion_min_trans", 0.25))
            min_trans = max(1e-3, min(1.0, min_trans))
            xyz = gaussians.get_xyz
            xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
            train_cams_local = scene.getTrainCameras()

            accum_depth = torch.zeros(xyz.shape[0], device="cuda")
            accum_count = torch.zeros(xyz.shape[0], device="cuda")

            sample_cams = train_cams_local
            if len(train_cams_local) > 20:
                sample_indices = np.random.choice(len(train_cams_local), 20, replace=False)
                sample_cams = [train_cams_local[int(i)] for i in sample_indices]

            for cam in sample_cams:
                w2c = cam.world_view_transform.cuda()
                xyz_cam = xyz_hom @ w2c
                depth = xyz_cam[:, 2]
                mask = depth > 0.0
                accum_depth[mask] += depth[mask]
                accum_count[mask] += 1.0

            avg_depth = accum_depth / accum_count.clamp_min(1.0)
            missing = (accum_count == 0.0)
            if missing.any():
                avg_depth[missing] = avg_depth[~missing].mean() if (~missing).any() else 1.0

            depth_norm = (avg_depth / scene_depth_max).clamp(0.0, 1.0)

            phys = physics_planes.query(depth_norm)
            beta_D = phys["beta_D"] # (N, 3)
            beta_B = phys["beta_B"] # (3,)
            B_inf = phys["B_inf"] # (3,)
            depth_scale = phys.get("depth_scale", torch.as_tensor(3.0, device="cuda"))
            depth_physical = depth_norm * depth_scale

            trans_D = torch.exp(-beta_D * depth_physical.unsqueeze(-1))
            trans_B = torch.exp(-beta_B * depth_physical.unsqueeze(-1))
            backscatter = B_inf * (1.0 - trans_B)

            # features_dc has shape (N, 1, 3)
            rgb_degraded = SH2RGB(gaussians._features_dc.squeeze(1)) # (N, 3)
            rgb_clean_full = (
                rgb_degraded - invert_blend * backscatter
            ) / trans_D.clamp_min(min_trans)
            rgb_clean = ((1.0 - invert_blend) * rgb_degraded
                         + invert_blend * rgb_clean_full)
            rgb_clean = rgb_clean.clamp(0.0, 1.0)

            new_features_dc = RGB2SH(rgb_clean).unsqueeze(1)
            gaussians._features_dc.copy_(new_features_dc)

            # Also scale features_rest (the higher SH coefficients) by transmission
            if hasattr(gaussians, "_features_rest") and gaussians._features_rest.numel() > 0:
                sh_scale = ((1.0 - invert_blend)
                            + invert_blend / trans_D.clamp_min(min_trans))
                new_features_rest = gaussians._features_rest * sh_scale.unsqueeze(1)
                gaussians._features_rest.copy_(new_features_rest.clamp(-2.0, 2.0))

            # Reset optimizer momentum states for features_dc and features_rest
            for group in gaussians.optimizer.param_groups:
                name = group.get("name", "")
                if name in ("f_dc", "f_rest"):
                    param = group["params"][0]
                    state = gaussians.optimizer.state.get(param, None)
                    if state is not None:
                        if "exp_avg" in state:
                            state["exp_avg"].zero_()
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"].zero_()
            print(f"[UW] Color correction done with blend={invert_blend:.2f}, "
                  f"min_trans={min_trans:.2f}; optimizer states reset.")

    # FINE STAGE — physics_planes RESTORED (was commented out)
    scene_reconstruction(
        dataset, opt, hyper, pipe,
        testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from,
        gaussians, scene, "fine", tb_writer,
        opt.iterations, timer,
        physics_planes=physics_planes,
        scene_bbox_min=scene_bbox_min,
        scene_bbox_max=scene_bbox_max,
        scene_depth_max=scene_depth_max,
        args=args,
        live_viewer=live_viewer,
    )

# basically unnecessary to bother about with respect to training, just meant for logging
def prepare_output_and_logger(expname):
    if not args.model_path:
        args.model_path = os.path.join("./output/", expname)
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        try:
            tb_writer = SummaryWriter(args.model_path)
            print(f"[TB] writer created at {args.model_path}")
        except Exception as e:
            print(f"[TB] failed to create writer: {e}")
    else:
        print("TensorBoard not found.")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                     testing_iterations, scene, renderFunc, renderArgs,
                     stage, dataset_type, renderKwargs=None):
    renderKwargs = renderKwargs or {}
    if tb_writer:
        tb_writer.add_scalar(f"{stage}/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{stage}/total_loss", loss.item(), iteration)
        tb_writer.add_scalar(f"{stage}/iter_time", elapsed, iteration)
        if iteration % 100 == 0:
            tb_writer.flush()

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        configs = (
            {"name": "test",
              "cameras": [scene.getTestCameras()[i % len(scene.getTestCameras())]
                          for i in range(10, 5000, 299)]},
            {"name": "train",
              "cameras": [scene.getTrainCameras()[i % len(scene.getTrainCameras())]
                          for i in range(10, 5000, 299)]},
        )
        for cfg in configs:
            if not cfg["cameras"]:
                continue
            l1_test = psnr_test = 0.0
            for idx, viewpoint in enumerate(cfg["cameras"]):
                image = torch.clamp(
                    renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                stage=stage, cam_type=dataset_type,
                                **renderKwargs)["render"], 0.0, 1.0)
                if dataset_type == "PanopticSports":
                    gt_image = torch.clamp(viewpoint["image"].cuda(), 0.0, 1.0)
                else:
                    gt_image = torch.clamp(
                        viewpoint.original_image.cuda(), 0.0, 1.0)
                if tb_writer and idx < 5:
                    tb_writer.add_images(
                        f"{stage}/{cfg['name']}_view_{viewpoint.image_name}/render",
                        image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            f"{stage}/{cfg['name']}_view_{viewpoint.image_name}/gt",
                            gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            n = len(cfg["cameras"])
            print(f"\n[ITER {iteration}] Evaluating {cfg['name']}: "
                  f"L1={l1_test/n:.4f}  PSNR={psnr_test/n:.2f}")
            if tb_writer:
                tb_writer.add_scalar(f"{stage}/{cfg['name']}/l1",
                                       l1_test / n, iteration)
                tb_writer.add_scalar(f"{stage}/{cfg['name']}/psnr",
                                       psnr_test / n, iteration)
                tb_writer.flush()
        torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="PhysSplit-4DGS training")
    setup_seed(6666)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6013)
    parser.add_argument("--viewer_port", type=int, default=7007,
                        help="Port for the viser live web viewer (default 7007)")
    parser.add_argument("--viewer_update_every", type=int, default=200,
                        help="Push to viewer every N fine iterations (default 200)")
    parser.add_argument("--disable_viewer", action="store_true", default=False,
                        help="Disable the viser live viewer")
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                          default=[3000, 7000, 12000, 14000,
                                   16000, 18000, 20000, 25000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                          default=[12000, 14000, 16000,
                                   18000, 20000, 25000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                          default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="underwater/set1")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--stage", type=str, default="both",
                          choices=["coarse", "fine", "both"])

    parser.add_argument("--water_type", type=str, default="II")
    parser.add_argument("--uw_color_inversion", action="store_true", default=False)
    parser.add_argument("--uw_color_inversion_blend", type=float, default=0.1)
    parser.add_argument("--uw_color_inversion_min_trans", type=float, default=0.25)
    parser.add_argument("--uw_lambda_transient", type=float, default=0.03)
    parser.add_argument("--uw_lambda_transient_residual", type=float, default=0.02)
    parser.add_argument("--uw_lambda_sparse", type=float, default=0.0002)
    parser.add_argument("--uw_warmup_iters", type=int, default=3000)
    parser.add_argument("--uw_aux_ramp_iters", type=int, default=5000)
    parser.add_argument("--uw_aux_peak_iter", type=int, default=12000)
    parser.add_argument("--uw_aux_decay_end_iter", type=int, default=20000)
    parser.add_argument("--uw_aux_decay_min", type=float, default=0.35)
    parser.add_argument("--uw_aux_w_max", type=float, default=0.30,
                        help="Hard cap on uw_aux_w to prevent UW losses dominating photo loss")
    parser.add_argument("--uw_tau_motion", type=float, default=0.05)
    parser.add_argument("--uw_sigma_thresh", type=float, default=0.50)
    parser.add_argument("--uw_sigma_target_mean", type=float, default=0.08)
    parser.add_argument("--uw_sigma_lower_bound", type=float, default=0.03)
    parser.add_argument("--uw_sigma_upper_bound", type=float, default=0.20)
    parser.add_argument("--uw_sigma_binarize_weight", type=float, default=0.05)
    parser.add_argument("--uw_lr_sigma", type=float, default=1e-3)
    parser.add_argument("--uw_lr_planes", type=float, default=5e-4)
    parser.add_argument("--uw_lambda_dark", type=float, default=0.05)
    parser.add_argument("--uw_lambda_binf", type=float, default=0.01)
    parser.add_argument("--uw_lambda_backscatter_dark", type=float, default=0.01)
    parser.add_argument("--uw_lambda_depth_weighted", type=float, default=0.01)
    parser.add_argument("--uw_lambda_mono_depth", type=float, default=0.01)
    parser.add_argument("--uw_lambda_depth_tv", type=float, default=0.002)
    parser.add_argument("--uw_lambda_tone", type=float, default=0.01)
    parser.add_argument("--uw_lambda_uncertainty", type=float, default=0.005)
    parser.add_argument("--uw_lambda_exposure", type=float, default=0.005)
    parser.add_argument("--uw_lambda_gray", type=float, default=0.002)
    parser.add_argument("--uw_lambda_saturation", type=float, default=0.005)
    parser.add_argument("--uw_lambda_spectral", type=float, default=0.005)
    parser.add_argument("--uw_mono_depth_residual_clip", type=float, default=2.0)
    parser.add_argument("--uw_mono_depth_inverse", action="store_true",
                          default=False)
    parser.add_argument("--uw_depth_weight_gamma", type=float, default=1.0)
    parser.add_argument("--uw_tone_mu", type=float, default=10.0)
    parser.add_argument("--uw_uncertainty_sharpness", type=float, default=6.0)
    parser.add_argument("--uw_exposure_upper", type=float, default=0.92)
    parser.add_argument("--uw_saturation_thresh", type=float, default=0.98)
    parser.add_argument("--uw_photo_guard_start", type=float, default=0.08)
    parser.add_argument("--uw_photo_guard_stop", type=float, default=0.25)
    parser.add_argument("--uw_dark_threshold", type=float, default=0.18)
    parser.add_argument("--uw_dark_fraction_threshold", type=float, default=0.01)
    parser.add_argument("--uw_dark_min_weight", type=float, default=0.20)
    parser.add_argument("--uw_binf_dark_thresh", type=float, default=0.25)
    parser.add_argument("--uw_binf_min_depth", type=float, default=0.25)
    parser.add_argument("--uw_anchor_min_pixels", type=int, default=256)
    parser.add_argument("--uw_backscatter_dark_thresh", type=float, default=0.25)
    parser.add_argument("--uw_backscatter_min_depth", type=float, default=0.25)
    parser.add_argument("--uw_disable_fine_pruning", action="store_true",
                          default=True)
    parser.add_argument("--uw_disable_fine_opacity_reset", action="store_true",
                          default=True)
    parser.add_argument("--uw_min_gaussians", type=int, default=20000)
    parser.add_argument("--uw_max_prune_fraction", type=float, default=0.25)
    # New losses
    parser.add_argument("--uw_lambda_physics_ordering", type=float, default=0.05)
    parser.add_argument("--uw_lambda_sigma_residual", type=float, default=0.02)

    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        spec = importlib.util.spec_from_file_location("_cfg", args.configs)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mod)

    if not getattr(args, "source_path", ""):
        raise ValueError("source_path is empty. Pass -s Data/set4 for training.")
    args.source_path = os.path.abspath(args.source_path)
    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"source_path does not exist: {args.source_path}")
    if getattr(args, "model_path", ""):
        args.model_path = os.path.normpath(args.model_path)
    args.save_iterations = sorted(set(
        int(i) for i in list(args.save_iterations) + [args.iterations]
        if int(i) <= int(args.iterations)))
    args.test_iterations = sorted(set(
        int(i) for i in args.test_iterations
        if int(i) <= int(args.iterations)))

    print("Optimising:", args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), hp.extract(args), op.extract(args),
        pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from, args.expname,
        args=args,
    )

    print("\nTraining complete.")
