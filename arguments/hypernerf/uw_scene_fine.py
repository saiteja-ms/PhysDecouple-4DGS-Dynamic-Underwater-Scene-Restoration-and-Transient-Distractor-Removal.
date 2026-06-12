class ModelParams:        # overridden by --model_path
    white_background = False
    sh_degree = 3
    images = "images"
    resolution = -1
    data_device = "cuda"
    eval = True

class OptimizationParams:
    iterations       = 20000
    coarse_iterations = 4000
    batch_size       = 1
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps  = 20000
    deformation_lr_init    = 0.00005
    deformation_lr_final   = 0.000005
    deformation_lr_delay_mult = 0.01
    grid_lr_init  = 0.0016
    grid_lr_final = 0.000016
    feature_lr    = 0.0025
    opacity_lr    = 0.05
    scaling_lr    = 0.005
    rotation_lr   = 0.001
    lambda_dssim  = 0.2
    densification_interval  = 100
    opacity_reset_interval  = 3000
    densify_from_iter       = 500
    densify_until_iter      = 15000
    densify_grad_threshold_coarse     = 0.00005
    densify_grad_threshold_fine_init  = 0.00005
    densify_grad_threshold_after      = 0.00005
    pruning_from_iter  = 500
    pruning_interval   = 100
    opacity_threshold_coarse     = 0.005
    opacity_threshold_fine_init  = 0.005
    opacity_threshold_fine_after = 0.005
    percent_dense = 0.01

class PipelineParams:
    convert_SHs_python   = False
    compute_cov3D_python = False
    debug = False

class ModelHiddenParams:
    net_width = 64
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
        "resolution": [128, 128, 128,50],
    }

    multires = [1, 2]

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

# class ModelHiddenParams:
#     net_width        = 64
#     defor_depth      = 1
#     weightdecay      = 0.0
#     grid_pe          = 0
#     skips            = []
#     multires         = [1, 2, 4, 8]
#     no_grid          = False
#     grid_type        = "plane_net"
#     no_ds            = False
#     no_dr            = False
#     no_do            = True
#     no_dc            = True
#     render_process   = False
#     static_iteration = 0
#     min_e            = 10
#     max_e            = 400
#     min_l            = 4
#     max_l            = 30