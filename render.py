"""
render.py — PhysSplit-4DGS evaluation

Updated for new architecture:
    - PhysicsPlanes no longer takes depth_res/time_res
    - Old checkpoint format incompatible; falls back to Jerlov priors if so
"""
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import (ModelParams, PipelineParams,
                        get_combined_args, ModelHiddenParams)
from scene.gaussian_model_uw import UnderwaterGaussianModel
from scene.physics_planes import PhysicsPlanes
from gaussian_renderer.renderer_uw import render_uw, render_clean
from gaussian_renderer import render as render_orig
import importlib.util
from time import time
import concurrent.futures


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except Exception:
            return count, False
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status.result()[1] is False:
            write_image(image_list[index], index, path)


to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def _as_chw_image(t):
    if t is None:
        return None
    t = t.detach().clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    return t


def estimate_scene_depth_max(scene, gaussians, pipeline, background,
                             physics_planes=None, n_samples=8,
                             mono_depth_inverse=False):
    train_cams = scene.getTrainCameras()
    if len(train_cams) == 0:
        return 1.0
    indices = np.random.choice(len(train_cams),
                                 size=min(n_samples, len(train_cams)),
                                 replace=False)
    sample_depths = []
    for idx in indices:
        try:
            cam = train_cams[int(idx)]
            if physics_planes is not None:
                pkg = render_uw(
                    cam, gaussians, pipeline, background,
                    stage="fine", cam_type=scene.dataset_type,
                    physics_planes=physics_planes,
                    scene_depth_max=None,
                    mono_depth_inverse=mono_depth_inverse,
                )
                d = pkg.get("depth_expected", None)
                opacity = pkg.get("accum_opacity", None)
                if d is not None and opacity is not None:
                    d = d[opacity > 0.05]
            else:
                pkg = render_orig(cam, gaussians, pipeline, background,
                                    stage="fine", cam_type=scene.dataset_type)
                d = pkg.get("depth", pkg.get("render_depth", None))
            if d is not None:
                if d.numel() > 0:
                    sample_depths.append(torch.quantile(d.float(), 0.95).item())
        except Exception:
            continue
    if not sample_depths:
        return 1.0
    sample_depths = sorted(sample_depths)
    return sample_depths[int(0.95 * (len(sample_depths) - 1))]


def render_set(model_path, name, iteration, views, gaussians, pipeline,
                background, cam_type, physics_planes=None,
                render_clean_mode=False, sigma_thresh=0.5,
                scene_depth_max=None, mono_depth_inverse=False,
                motion_thresh=None):

    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    clean_path = os.path.join(model_path, name, f"ours_{iteration}",
                              "renders_clean")
    clean_tr_path = os.path.join(model_path, name, f"ours_{iteration}",
                                 "renders_clean_transient_removed"
                                 if motion_thresh is None else
                                 "renders_clean_static")
    _suffix = "transient_removed" if motion_thresh is None else "static"
    render_tr_path = os.path.join(model_path, name, f"ours_{iteration}",
                                  f"renders_{_suffix}")
    depth_path = os.path.join(model_path, name, f"ours_{iteration}", "depth_norm")
    sigma_path = os.path.join(model_path, name, f"ours_{iteration}", "sigma")
    backscatter_path = os.path.join(model_path, name, f"ours_{iteration}",
                                    "backscatter")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(clean_path, exist_ok=True)
    makedirs(clean_tr_path, exist_ok=True)
    makedirs(render_tr_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(sigma_path, exist_ok=True)
    makedirs(backscatter_path, exist_ok=True)

    render_images = []
    gt_list = []
    render_list = []
    clean_list = []
    clean_tr_list = []
    render_tr_list = []
    depth_list = []
    sigma_list = []
    backscatter_list = []

    print("point nums:", gaussians._xyz.shape[0])
    print(f"[EVAL] using scene_depth_max = {scene_depth_max}")
    if len(views) == 0:
        print(f"[EVAL] no {name} views to render")
        return

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:
            time1 = time()

        scene_bbox_min = gaussians.get_xyz.detach().min(dim=0).values
        scene_bbox_max = gaussians.get_xyz.detach().max(dim=0).values

        pkg = render_uw(
            view, gaussians, pipeline, background,
            physics_planes=physics_planes,
            scene_bbox_min=scene_bbox_min,
            scene_bbox_max=scene_bbox_max,
            scene_depth_max=scene_depth_max,
            mono_depth_inverse=mono_depth_inverse,
            cam_type=cam_type,
            stage="fine",
        )

        pkg_tr = None
        if physics_planes is not None and hasattr(gaussians, "get_sigma"):
            pkg_tr = render_clean(
                view, gaussians, pipeline, background,
                physics_planes=physics_planes,
                sigma_thresh=sigma_thresh,
                scene_bbox_min=scene_bbox_min,
                scene_bbox_max=scene_bbox_max,
                scene_depth_max=scene_depth_max,
                mono_depth_inverse=mono_depth_inverse,
                cam_type=cam_type,
                stage="fine",
                motion_thresh=motion_thresh,
            )

        render = pkg.get("render_degraded", pkg["render"])
        render_list.append(render)

        clean_img = pkg.get("render_J", pkg.get("render_clean", None))
        if clean_img is not None:
            clean_list.append(clean_img.clamp(0, 1))

        if pkg_tr is not None:
            clean_tr_img = pkg_tr.get("render_J", pkg_tr.get("render_clean", None))
            if clean_tr_img is not None:
                clean_tr_list.append(clean_tr_img.clamp(0, 1))
            render_tr_img = pkg_tr.get("render_degraded", pkg_tr.get("render"))
            if render_tr_img is not None:
                render_tr_list.append(render_tr_img.clamp(0, 1))

        depth_img = _as_chw_image(pkg.get("depth_norm", None))
        if depth_img is not None:
            depth_list.append(depth_img)
        sigma_img = _as_chw_image(pkg.get("sigma_render", None))
        if sigma_img is not None:
            sigma_list.append(sigma_img)
        backscatter_img = pkg.get("render_backscatter", None)
        if backscatter_img is not None:
            backscatter_list.append(backscatter_img.clamp(0, 1))

        video_src = render
        if render_clean_mode:
            if clean_tr_list:
                video_src = clean_tr_list[-1]
            elif clean_list:
                video_src = clean_list[-1]
        render_images.append(to8b(video_src).transpose(1, 2, 0))

        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt = view['image'].cuda()
            gt_list.append(gt)

    time2 = time()
    print("FPS:", len(views) / max(time2 - time1, 1e-6))

    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    multithread_write(clean_list, clean_path)
    multithread_write(clean_tr_list, clean_tr_path)
    multithread_write(render_tr_list, render_tr_path)
    multithread_write(depth_list, depth_path)
    multithread_write(sigma_list, sigma_path)
    multithread_write(backscatter_list, backscatter_path)
    imageio.mimwrite(
        os.path.join(model_path, name, f"ours_{iteration}", 'video_rgb.mp4'),
        render_images, fps=30)


def render_sets(dataset, hyperparam, iteration, pipeline,
                 skip_train, skip_test, skip_video,
                 render_clean_mode=False, sigma_thresh=0.5,
                 water_type="II", mono_depth_inverse=False,
                 motion_thresh=None):

    with torch.no_grad():
        gaussians = UnderwaterGaussianModel(
            dataset.sh_degree,
            hyperparam,
            water_type=water_type,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        physics_planes = PhysicsPlanes(
            water_type=water_type,
        ).cuda()

        phys_path = os.path.join(
            dataset.model_path,
            f"physics_planes_fine_{scene.loaded_iter}.pth",
        )
        if os.path.exists(phys_path):
            try:
                physics_planes.load_state_dict(
                    torch.load(phys_path, map_location="cuda"))
                print(f"Loaded physics planes from {phys_path}")
            except Exception as e:
                print(f"[WARN] couldn't load physics_planes: {e}")
                print(f"[WARN] using Jerlov priors instead")
        else:
            print(f"[WARN] {phys_path} not found, using Jerlov priors")

        physics_planes.eval()

        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        depth_meta_path = os.path.join(dataset.model_path, "scene_depth_max.txt")
        if os.path.exists(depth_meta_path):
            with open(depth_meta_path) as f:
                scene_depth_max = float(f.read().strip())
            print(f"[EVAL] loaded scene_depth_max={scene_depth_max:.3f}")
        else:
            print(f"[EVAL] scene_depth_max.txt not found, re-estimating")
            scene_depth_max = estimate_scene_depth_max(
                scene, gaussians, pipeline, background,
                physics_planes=physics_planes, n_samples=8,
                mono_depth_inverse=mono_depth_inverse)
            print(f"[EVAL] estimated scene_depth_max={scene_depth_max:.3f}")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, cam_type, physics_planes,
                        render_clean_mode, sigma_thresh, scene_depth_max,
                        mono_depth_inverse, motion_thresh)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, cam_type, physics_planes,
                        render_clean_mode, sigma_thresh, scene_depth_max,
                        mono_depth_inverse, motion_thresh)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter,
                        scene.getVideoCameras(), gaussians, pipeline,
                        background, cam_type, physics_planes,
                        render_clean_mode, sigma_thresh, scene_depth_max,
                        mono_depth_inverse, motion_thresh)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--render_clean", action="store_true")
    parser.add_argument("--sigma_thresh", type=float, default=0.5)
    parser.add_argument("--motion_thresh", type=float, default=None,
                        help="Also suppress persistent-dynamic Gaussians whose "
                             "canonical displacement exceeds this (static scene "
                             "extraction; e.g. 0.02)")
    parser.add_argument("--water_type", type=str, default="II")
    parser.add_argument("--uw_mono_depth_inverse", action="store_true",
                        default=False)

    args = get_combined_args(parser)
    print("Rendering", args.model_path)

    if args.configs:
        spec = importlib.util.spec_from_file_location("_uw_cfg", args.configs)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mod)
    
    render_defaults = {
        "iteration": -1,
        "skip_train": False,
        "skip_test": False,
        "skip_video": False,
        "render_clean": False,
        "sigma_thresh": 0.50,
        "water_type": "II",
        "uw_mono_depth_inverse": True,
        "quiet": False,
    }

    for k, v in render_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not getattr(args, "source_path", ""):
        raise ValueError(
            "source_path is empty. Re-run render with -s Data/set4, or train "
            "from a run whose cfg_args saved source_path.")
    args.source_path = os.path.abspath(args.source_path)
    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"source_path does not exist: {args.source_path}")

    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        hyperparam.extract(args),
        args.iteration,
        pipeline.extract(args),
        getattr(args, "skip_train", False),
        getattr(args, "skip_test", False),
        getattr(args, "skip_video", False),
        render_clean_mode=args.render_clean,
        sigma_thresh=args.sigma_thresh,
        water_type=args.water_type,
        mono_depth_inverse=getattr(args, "uw_mono_depth_inverse", False),
        motion_thresh=args.motion_thresh,
    )
