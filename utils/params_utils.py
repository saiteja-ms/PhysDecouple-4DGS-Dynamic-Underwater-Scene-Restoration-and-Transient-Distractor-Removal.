def merge_hparams(args, config):
    sections = [
        "OptimizationParams",
        "ModelHiddenParams",
        "ModelParams",
        "PipelineParams",
    ]

    # CLI path/output args should win over config defaults.
    cli_protected = {
        "source_path",
        "model_path",
        "expname",
        "configs",
        "ip",
        "port",
    }

    for section_name in sections:
        if not hasattr(config, section_name):
            continue

        section = getattr(config, section_name)

        if isinstance(section, dict):
            items = section.items()
        else:
            items = vars(section).items()

        for key, value in items:
            if key.startswith("__"):
                continue

            if key in cli_protected:
                continue

            if hasattr(args, key):
                setattr(args, key, value)

    return args


# def merge_hparams(args, config):
#     params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
#     for param in params:
#         if param in config.keys():
#             for key, value in config[param].items():
#                 if hasattr(args, key):
#                     setattr(args, key, value)

#     return args