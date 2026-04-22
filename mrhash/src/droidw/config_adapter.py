"""
Config adapter for DROID-W integration.

Converts the unified integrated config (YAML) into the format
expected by DROID-W's SLAM pipeline.
"""

import yaml


def load_config(path, default_path=None):
    """Load config file with inheritance support (from DROID-W)."""
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)
    return cfg


def save_config(cfg, path):
    with open(path, "w+") as fp:
        yaml.dump(cfg, fp)


def update_recursive(dict1, dict2):
    """Update dict1 recursively with dict2."""
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def build_droidw_config(integrated_cfg):
    """
    Extract and build the DROID-W specific configuration from
    the unified integrated config.

    Args:
        integrated_cfg: dict, the full integrated config

    Returns:
        droidw_cfg: dict, config compatible with DROID-W SLAM
    """
    dw = integrated_cfg.get("droidw", {})
    sensor = integrated_cfg.get("sensor", {})

    # Build DROID-W compatible config
    droidw_cfg = {
        "verbose": dw.get("verbose", True),
        "gui": False,
        "stride": dw.get("stride", 1),
        "max_frames": dw.get("max_frames", -1),
        "setup_seed": dw.get("setup_seed", 43),
        "fast_mode": dw.get("fast_mode", True),
        "device": dw.get("device", "cuda:0"),
        "save_gt_poses": dw.get("save_gt_poses", True),
        "droidvis": False,
        "debug": False,
        "dataset": "tumrgbd",
        "scene": integrated_cfg.get("scene", "integrated_run"),
        "traj_filler": dw.get("traj_filler", {"use_dino_features": False}),
        "mapping": {
            "enable": False,  # We use dynamic_3dgs for mapping, not DROID-W's GS
            "online_plotting": False,
            "full_resolution": False,
            "final_refine_iters": 3000,
            "eval_before_final_ba": False,
            "deform_gaussians": False,
            "pcd_downsample": 32,
            "pcd_downsample_init": 16,
            "adaptive_pointsize": True,
            "point_size": 0.05,
            "Training": dw.get("mapping_training", {
                "ssim_loss": True,
                "alpha": 0.8,
                "init_itr_num": 500,
                "init_gaussian_update": 100,
                "init_gaussian_reset": 500,
                "init_gaussian_th": 0.005,
                "init_gaussian_extent": 30,
                "mapping_itr_num": 50,
                "gaussian_update_every": 1500,
                "gaussian_update_offset": 500,
                "gaussian_th": 0.7,
                "gaussian_extent": 1.0,
                "gaussian_reset": 20001,
                "size_threshold": 20,
                "window_size": 10,
                "edge_threshold": 4,
                "rgb_boundary_threshold": 0.01,
                "spherical_harmonics": False,
                "lr": {
                    "cam_rot_delta": 0.003,
                    "cam_trans_delta": 0.001,
                },
            }),
            "opt_params": dw.get("mapping_opt_params", {
                "position_lr_init": 0.00016,
                "position_lr_final": 0.0000016,
                "position_lr_delay_mult": 0.01,
                "position_lr_max_steps": 30000,
                "feature_lr": 0.0025,
                "opacity_lr": 0.05,
                "scaling_lr": 0.001,
                "rotation_lr": 0.001,
                "percent_dense": 0.01,
                "lambda_dssim": 0.2,
                "densification_interval": 100,
                "opacity_reset_interval": 3000,
                "densify_until_iter": 15000,
                "densify_grad_threshold": 0.0002,
            }),
            "model_params": {"sh_degree": 0},
            "pipeline_params": {
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
            },
            "uncertainty_params": dw.get("mapping_uncertainty_params", {
                "activate": True,
                "vis_uncertainty_online": False,
                "mapping_loss_type": "normalized_l1",
                "train_frac_fix": 0.3,
                "ssim_window_size": 7,
                "ssim_median_filter_size": 5,
                "reg_stride": 2,
                "opacity_th_for_uncer_loss": 0.9,
                "uncer_depth_mult": 0.2,
            }),
        },
        "tracking": dw.get("tracking", {
            "pretrained": "pretrained/droid.pth",
            "buffer": 350,
            "beta": 0.75,
            "warmup": 12,
            "max_age": 50,
            "mono_thres": 0.1,
            "motion_filter": {"thresh": 3.0},
            "multiview_filter": {"thresh": 0.01, "visible_num": 2},
            "frontend": {
                "enable_loop": False,
                "enable_online_ba": False,
                "keyframe_thresh": 3.0,
                "thresh": 16.0,
                "window": 25,
                "radius": 2,
                "nms": 1,
                "max_factors": 75,
                "enable_opt_dyn_mask": True,
            },
            "backend": {
                "final_ba": False,  # Disabled for integration
                "ba_freq": 20,
                "thresh": 25.0,
                "radius": 1,
                "nms": 5,
                "loop_window": 25,
                "loop_thresh": 25.0,
                "loop_radius": 1,
                "loop_nms": 10,
                "metric_depth_reg": True,
                "normalize": False,
            },
            "uncertainty_params": {
                "feature_dim": 384,
                "activate": True,
                "visualize": False,
                "gamma_data": 0.1,
                "gamma_prior": 0.05,
                "gamma_depth": 0.001,
                "lr": 0.1,
                "weight_decay": 0.002,
                "gba_lr": 0.1,
                "gba_weight_decay": 0.002,
                "enable_affine_transform": True,
                "enable_bidirectional_uncer": False,
            },
            "force_keyframe_every_n_frames": 9,
            "nb_ref_frame_metric_depth_filtering": 6,
        }),
        "cam": {
            "H": sensor.get("resolution", [640, 480])[1],
            "W": sensor.get("resolution", [640, 480])[0],
            "fx": sensor.get("intrinsics", [535.4, 539.2, 320.1, 247.6])[0],
            "fy": sensor.get("intrinsics", [535.4, 539.2, 320.1, 247.6])[1],
            "cx": sensor.get("intrinsics", [535.4, 539.2, 320.1, 247.6])[2],
            "cy": sensor.get("intrinsics", [535.4, 539.2, 320.1, 247.6])[3],
            "png_depth_scale": sensor.get("depth_scaling", 5000.0),
            "H_edge": dw.get("cam_H_edge", 8),
            "W_edge": dw.get("cam_W_edge", 8),
            "H_out": dw.get("cam_H_out", 384),
            "W_out": dw.get("cam_W_out", 512),
        },
        "mono_prior": dw.get("mono_prior", {
            "depth": "metric3d_vit_large",
            "save_depth": False,
            "feature_extractor": "dinov2_reg_small_fine",
            "save_feature": False,
        }),
        "data": {
            "output": integrated_cfg.get("results_path", "/tmp/droidw_output"),
            "input_folder": integrated_cfg.get("data_path", ""),
        },
    }

    # Override tracking config from integrated config if provided
    if "tracking" in dw:
        update_recursive(droidw_cfg["tracking"], dw["tracking"])

    return droidw_cfg
