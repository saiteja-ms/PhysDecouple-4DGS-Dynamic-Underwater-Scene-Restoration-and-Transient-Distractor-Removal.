import sys
import os
import argparse
from argparse import ArgumentParser, Namespace
import importlib.util

# Add the directory containing this script to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from render import render_sets

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
    parser.add_argument("--motion_thresh", type=float, default=None)
    parser.add_argument("--water_type", type=str, default="II")
    parser.add_argument("--uw_mono_depth_inverse", action="store_true", default=False)

    args = get_combined_args(parser)
    print("Rendering (patched)", args.model_path)

    # Cache original args from cfg_args to re-apply after config loading
    cfgfilepath = os.path.join(args.model_path, "cfg_args")
    args_cfgfile = None
    if os.path.exists(cfgfilepath):
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
            try:
                args_cfgfile = eval(cfgfile_string, {"Namespace": Namespace})
            except Exception as e:
                print(f"[WARN] failed to eval cfg_args: {e}")

    if args.configs:
        spec = importlib.util.spec_from_file_location("_uw_cfg", args.configs)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mod)
    
    # Re-apply training hyperparameters to avoid mismatch
    if args_cfgfile is not None:
        print("[DEBUG] Re-applying training args from cfg_args:")
        for k, v in vars(args_cfgfile).items():
            print(f"  {k} = {v}")
            setattr(args, k, v)
    
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
        "motion_thresh": None,
    }

    for k, v in render_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not getattr(args, "source_path", ""):
        raise ValueError("source_path is empty.")
    args.source_path = os.path.abspath(args.source_path)

    extracted_hyper = hyperparam.extract(args)
    print("[DEBUG] extracted_hyper.net_width =", getattr(extracted_hyper, "net_width", None))
    print("[DEBUG] extracted_hyper.multires =", getattr(extracted_hyper, "multires", None))
    print("[DEBUG] args.net_width =", getattr(args, "net_width", None))
    print("[DEBUG] args.multires =", getattr(args, "multires", None))

    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        extracted_hyper,
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
