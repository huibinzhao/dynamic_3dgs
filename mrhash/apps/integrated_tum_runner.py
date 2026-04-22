"""
Integrated Runner: DROID-W Visual Odometry + Dynamic 3DGS Mapping

This is the unified entry point for the integrated pipeline that:
1. Loads TUM RGB-D data with a single dataloader
2. Runs DROID-W tracking to estimate camera poses (or uses GT poses)
3. Runs Dynamic 3DGS (MRHash) mapping with the estimated/GT poses
4. Outputs all results to a single directory

Runs under the dynamic_3dgs conda environment:
    conda activate dynamic_3dgs
    cd /home/robin/dynamic_3dgs
    python mrhash/apps/integrated_tum_runner.py [config_path]

Usage:
    python mrhash/apps/integrated_tum_runner.py mrhash/configurations/tum_integrated.cfg
"""

import shutil
import sys
import os
import time
from pathlib import Path

# --- Environment setup (must be before other imports) ---
# Ensure conda env lib is first in LD_LIBRARY_PATH to avoid
# ROS libtiff/libjpeg conflicts with opencv
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _conda_lib not in _ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = _conda_lib + (":" + _ld if _ld else "")

import cv2
import numpy as np
import typer
import yaml
from typing_extensions import Annotated
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from tqdm import tqdm

# Ensure the apps directory is in the path for relative imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Add mrhash/src to path so "import droidw" resolves to mrhash/src/droidw
_MRHASH_SRC = str(SCRIPT_DIR.parent / "src")
if _MRHASH_SRC not in sys.path:
    sys.path.insert(0, _MRHASH_SRC)

from utils.camera import Camera, CameraModel
from utils.tum_unified_reader import TUMUnifiedReader

console = Console()


def print_banner():
    """Print startup banner."""
    banner = (
        "[bold cyan]╔══════════════════════════════════════════════════╗[/]\n"
        "[bold cyan]║[/]  [bold white]DROID-W + Dynamic 3DGS Integrated Pipeline[/]    [bold cyan]║[/]\n"
        "[bold cyan]║[/]  [dim]Visual Odometry → Dynamic 3D Gaussian Mapping[/] [bold cyan]║[/]\n"
        "[bold cyan]╚══════════════════════════════════════════════════╝[/]"
    )
    console.print(banner)


def print_config_summary(data_cf, use_gt_pose):
    """Print configuration summary table."""
    table = Table(title="Configuration Summary", box=box.ROUNDED)
    table.add_column("Section", style="cyan", width=20)
    table.add_column("Parameter", style="yellow", width=30)
    table.add_column("Value", style="white", width=30)

    # Pose source
    table.add_row(
        "Pipeline",
        "Pose Source",
        "[green]GT Pose[/]" if use_gt_pose else "[blue]DROID-W Odometry[/]",
    )
    table.add_row("Pipeline", "Data Path", str(data_cf["data_path"]))
    table.add_row("Pipeline", "Results Path", str(data_cf["results_path"]))

    # Sensor
    sensor = data_cf["sensor"]
    table.add_row("Sensor", "Resolution", str(sensor["resolution"]))
    table.add_row("Sensor", "Intrinsics", str(sensor["intrinsics"]))
    table.add_row("Sensor", "Depth Range", f'{sensor["min_depth"]} - {sensor["max_depth"]} m')
    table.add_row("Sensor", "Depth Scaling", str(sensor["depth_scaling"]))

    # Map
    m = data_cf["map"]
    table.add_row("3DGS Map", "SDF Truncation", str(m["sdf_truncation"]))
    table.add_row("3DGS Map", "Voxel Size", str(m["virtual_voxel_size"]))
    table.add_row("3DGS Map", "Dynamic Detection", str(m.get("dynamic_detection", False)))
    table.add_row("3DGS Map", "Dynamic Method", str(m.get("dynamic_method", "none")))

    # DROID-W
    if not use_gt_pose:
        dw = data_cf.get("droidw", {})
        tracking = dw.get("tracking", {})
        table.add_row("DROID-W", "Device", str(dw.get("device", "cuda:0")))
        table.add_row("DROID-W", "Buffer Size", str(tracking.get("buffer", 350)))
        table.add_row(
            "DROID-W",
            "Motion Filter Thresh",
            str(tracking.get("motion_filter", {}).get("thresh", 3.0)),
        )
        table.add_row(
            "DROID-W",
            "Pretrained Model",
            str(tracking.get("pretrained", "N/A")),
        )

    console.print(table)


def print_section(title, color="cyan"):
    """Print a section divider."""
    console.print(f"\n[bold {color}]{'═' * 60}[/]")
    console.print(f"[bold {color}]  {title}[/]")
    console.print(f"[bold {color}]{'═' * 60}[/]\n")


def run_droidw_tracking(data_cf, reader):
    """
    Run DROID-W visual odometry to estimate camera poses.

    Args:
        data_cf: full config dict
        reader: TUMUnifiedReader instance

    Returns:
        dict with estimated poses (translations and quaternions for all frames)
    """
    print_section("Phase 1: DROID-W Visual Odometry", "blue")

    # Setup DROID-W imports
    import droidw  # This adds DROID-W to sys.path
    from droidw.config_adapter import build_droidw_config
    from droidw.dataset_wrapper import DroidWTUMDataset
    from droidw.slam_wrapper import DroidWSLAMWrapper, IntegratedPrinter

    # Build DROID-W config from integrated config
    droidw_cfg = build_droidw_config(data_cf)

    console.print("[blue][DROID-W][/] Building dataset for tracking...")

    # Create DROID-W compatible dataset
    color_paths = reader.get_color_paths()
    depth_paths = reader.get_depth_paths()
    gt_poses = reader.get_gt_poses_4x4()

    droidw_dataset = DroidWTUMDataset(
        cfg=droidw_cfg,
        color_paths=color_paths,
        depth_paths=depth_paths,
        gt_poses_4x4=gt_poses,
        device=droidw_cfg["device"],
    )

    # Save GT poses if available
    if gt_poses is not None and droidw_cfg.get("save_gt_poses", True):
        output_dir = Path(data_cf["results_path"]) / "droidw"
        output_dir.mkdir(parents=True, exist_ok=True)
        droidw_dataset.save_gt_poses(
            str(output_dir / "gt_poses.txt"), gt_poses
        )
        console.print(f"[blue][DROID-W][/] GT poses saved to {output_dir / 'gt_poses.txt'}")

    # Create printer and SLAM wrapper
    printer = IntegratedPrinter(len(droidw_dataset))

    slam_wrapper = DroidWSLAMWrapper(
        droidw_cfg, droidw_dataset, printer
    )

    # Run tracking
    console.print(
        f"[blue][DROID-W][/] Starting tracking on {len(droidw_dataset)} frames..."
    )
    pose_data = slam_wrapper.run_tracking()

    console.print(
        f"[blue][DROID-W][/] Tracking complete: "
        f"{pose_data['num_keyframes']} keyframes from {pose_data['num_frames']} frames"
    )

    # Save estimated poses
    output_dir = Path(data_cf["results_path"]) / "droidw"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_dir / "estimated_poses.npz"),
        keyframe_indices=pose_data["keyframe_indices"],
        all_translations=pose_data["all_translations"],
        all_quaternions=pose_data["all_quaternions"],
        all_poses_4x4=pose_data["all_poses_4x4"],
    )
    console.print(
        f"[blue][DROID-W][/] Estimated poses saved to {output_dir / 'estimated_poses.npz'}"
    )

    return pose_data, slam_wrapper


def run_dynamic_3dgs_mapping(data_cf, reader, pose_data, use_gt_pose, timestamp, slam_wrapper=None):
    """
    Run Dynamic 3DGS (MRHash) mapping with given poses.

    Args:
        data_cf: full config dict
        reader: TUMUnifiedReader instance
        pose_data: dict with estimated poses (or None if using GT)
        use_gt_pose: whether to use GT poses
        timestamp: string timestamp for output files
    """
    print_section("Phase 2: Dynamic 3DGS Mapping", "green")

    from mrhash.src.pygeowrapper import GeoWrapper

    results_dir = Path(data_cf["results_path"])

    # Load map config
    sdf_truncation = data_cf["map"]["sdf_truncation"]
    sdf_truncation_scale = data_cf["map"]["sdf_truncation_scale"]
    integration_weight_sample = data_cf["map"]["integration_weight_sample"]
    virtual_voxel_size = data_cf["map"]["virtual_voxel_size"]
    n_frames_invalidate_voxels = data_cf["map"]["n_frames_invalidate_voxels"]
    dynamic_detection = data_cf["map"].get("dynamic_detection", False)
    dynamic_method = data_cf["map"].get("dynamic_method", "tsdf_residual")
    # dynamic_detection is master switch; dynamic_method selects the approach
    # Enable C++ TSDF residual detection only for tsdf_residual method
    enable_tsdf_residual = dynamic_detection and dynamic_method == "tsdf_residual"
    dynamic_erosion_size = data_cf["map"].get("dynamic_erosion_size", 15)
    dynamic_dilation_size = data_cf["map"].get("dynamic_dilation_size", 10)
    dynamic_flood_threshold = data_cf["map"].get("dynamic_flood_threshold", 0.007)
    save_dynamic_mask = data_cf["map"].get("save_dynamic_mask", False)
    gs_only_dynamic_frames = data_cf["map"].get("gs_only_dynamic_frames", False)
    gs_visualize = data_cf["map"].get("gs_visualize", False)

    voxel_extents_scale = data_cf["streamer"]["voxel_extents_scale"]

    marching_cubes_threshold = data_cf["mesh"]["marching_cubes_threshold"]
    min_weight_threshold = data_cf["mesh"]["min_weight_threshold"]
    sdf_var_threshold = data_cf["mesh"]["sdf_var_threshold"]
    vertices_merging_threshold = data_cf["mesh"]["vertices_merging_threshold"]

    gs_optimization_param_path = data_cf.get("gs_optimization_param_path", "")

    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = data_cf["sensor"]["intrinsics"][0]
    K[1, 1] = data_cf["sensor"]["intrinsics"][1]
    K[0, 2] = data_cf["sensor"]["intrinsics"][2]
    K[1, 2] = data_cf["sensor"]["intrinsics"][3]
    K[2, 2] = 1

    img_rows = data_cf["sensor"]["resolution"][1]
    img_cols = data_cf["sensor"]["resolution"][0]
    min_depth = data_cf["sensor"]["min_depth"]
    max_depth = data_cf["sensor"]["max_depth"]

    end_frame = data_cf["end_frame"] if data_cf["end_frame"] != -1 else len(reader) + 1

    # Check dynamic method
    dynamic_detection = data_cf["map"].get("dynamic_detection", False)
    dynamic_method = data_cf["map"].get("dynamic_method", "tsdf_residual")
    use_droidw_uncertainty = (dynamic_detection and dynamic_method == "droidw_uncertainty" and slam_wrapper is not None)
    uncertainty_threshold = data_cf["map"].get("droidw_uncertainty_threshold", 0.9)
    uncertainty_dilation = data_cf["map"].get("droidw_uncertainty_dilation", 10)
    map_keyframes_only = data_cf["map"].get("map_keyframes_only", False)

    # Build keyframe index set for filtering
    keyframe_set = None
    if map_keyframes_only and pose_data is not None and "keyframe_indices" in pose_data:
        keyframe_set = set(int(k) for k in pose_data["keyframe_indices"])
        console.print(f"[green][3DGS-MAP][/] Keyframe-only mapping: {len(keyframe_set)} keyframes")

    # Print 3DGS config info
    table = Table(title="Dynamic 3DGS Parameters", box=box.SIMPLE)
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="white")
    table.add_row("SDF Truncation", str(sdf_truncation))
    table.add_row("Voxel Size", str(virtual_voxel_size))
    table.add_row("Dynamic Method", dynamic_method)
    if use_droidw_uncertainty:
        table.add_row("Uncertainty Threshold", str(uncertainty_threshold))
        table.add_row("Uncertainty Dilation", str(uncertainty_dilation))
    table.add_row("Dynamic Detection (TSDF)", str(dynamic_detection))
    table.add_row("GS Visualize", str(gs_visualize))
    table.add_row("Num Frames", str(len(reader)))
    table.add_row("Pose Source", "GT" if use_gt_pose else "DROID-W Estimated")
    console.print(table)

    # Setup camera
    rgbd_camera = Camera(
        rows=img_rows,
        cols=img_cols,
        K=K,
        min_depth=min_depth,
        max_depth=max_depth,
        model=CameraModel.Pinhole,
    )

    # Initialize GeoWrapper
    geo_wrapper = GeoWrapper(
        sdf_truncation=sdf_truncation,
        sdf_truncation_scale=sdf_truncation_scale,
        integration_weight_sample=integration_weight_sample,
        virtual_voxel_size=virtual_voxel_size,
        n_frames_invalidate_voxels=n_frames_invalidate_voxels,
        voxel_extents_scale=voxel_extents_scale,
        viewer_active=False,
        marching_cubes_threshold=marching_cubes_threshold,
        min_weight_threshold=min_weight_threshold,
        sdf_var_threshold=sdf_var_threshold,
        gs_optimization_param_path=gs_optimization_param_path,
        vertices_merging_threshold=vertices_merging_threshold,
        projective_sdf=True,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    geo_wrapper.enableDynamicDetection(enable_tsdf_residual)
    geo_wrapper.setDynamicErosionSize(dynamic_erosion_size)
    geo_wrapper.setDynamicDilationSize(dynamic_dilation_size)
    geo_wrapper.setDynamicFloodThreshold(dynamic_flood_threshold)
    geo_wrapper.setGSOnlyDynamicFrames(gs_only_dynamic_frames)
    geo_wrapper.setGSVisualize(gs_visualize)

    # Configure mask saving
    if save_dynamic_mask:
        mask_dir = results_dir / "mrhash_mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        (mask_dir / "raw").mkdir(parents=True, exist_ok=True)
        geo_wrapper.setSaveDynamicMask(True)
        geo_wrapper.setMaskOutputPath(str(mask_dir))
        console.print(f"[green][3DGS-MAP][/] Saving masks to: {mask_dir}")

    geo_wrapper.setCamera(
        rgbd_camera.fx_,
        rgbd_camera.fy_,
        rgbd_camera.cx_,
        rgbd_camera.cy_,
        rgbd_camera.rows_,
        rgbd_camera.cols_,
        rgbd_camera.min_depth_,
        rgbd_camera.max_depth_,
        rgbd_camera.model_,
    )

    # Process frames
    console.print(
        f"[green][3DGS-MAP][/] Processing {min(len(reader), end_frame)} frames..."
    )

    for idx in tqdm(range(len(reader)), desc="[3DGS-MAP] Mapping"):
        # Skip non-keyframes when map_keyframes_only is enabled
        if keyframe_set is not None and idx not in keyframe_set:
            # Still visualize all frames: render from current pose without mapping
            if gs_visualize:
                if use_gt_pose:
                    _, translation, quat, _, vis_rgb = reader[idx]
                else:
                    _, _, _, _, vis_rgb = reader[idx]
                    translation = pose_data["all_translations"][idx]
                    quat = pose_data["all_quaternions"][idx]
                geo_wrapper.setCurrPose(translation, quat)
                geo_wrapper.GSRenderOnly()
                gt_bgr = cv2.cvtColor(vis_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                if geo_wrapper.hasGSRenderedImage():
                    rendered = np.array(geo_wrapper.getGSRenderedImage())
                    rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                    combined = np.hstack([gt_bgr, rendered_bgr])
                else:
                    combined = gt_bgr
                cv2.imshow("3DGS: GT (left) | Rendered (right)", combined)
                cv2.waitKey(1)
            continue

        if use_gt_pose:
            # Use GT pose from dataloader
            frame, translation, quat, depth_img, rgb_img = reader[idx]
        else:
            # Use DROID-W estimated pose
            frame, _, _, depth_img, rgb_img = reader[idx]
            translation = pose_data["all_translations"][idx]
            quat = pose_data["all_quaternions"][idx]

        if frame > end_frame:
            break

        # Get uncertainty mask if using droidw_uncertainty method
        mask = None
        if use_droidw_uncertainty:
            mask = slam_wrapper.get_uncertainty_mask(
                idx, img_rows, img_cols, uncertainty_threshold, uncertainty_dilation
            )

        geo_wrapper.setCurrPose(translation, quat)
        geo_wrapper.setDepthImage(depth_img)
        geo_wrapper.setRGBImage(rgb_img)
        if mask is not None:
            geo_wrapper.setExternalDynamicMask(mask)
        geo_wrapper.compute()

        # Show GS visualization if enabled
        if gs_visualize and geo_wrapper.hasGSRenderedImage():
            rendered = np.array(geo_wrapper.getGSRenderedImage())
            gt_display = rgb_img.astype(np.uint8)
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            gt_bgr = cv2.cvtColor(gt_display, cv2.COLOR_RGB2BGR)
            combined = np.hstack([gt_bgr, rendered_bgr])
            cv2.imshow("3DGS: GT (left) | Rendered (right)", combined)
            cv2.waitKey(1)

    if gs_visualize and geo_wrapper.hasGSRenderedImage():
        console.print(
            "[green][3DGS-MAP][/] Mapping complete. Press any key on the image window to close."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save outputs
    print_section("Saving Results", "yellow")

    geo_wrapper.streamAllOut()

    mesh_path = f"{results_dir}/mesh_{timestamp}.ply"
    geo_wrapper.extractMesh(mesh_path)
    console.print(f"[yellow][OUTPUT][/] Mesh: {mesh_path}")

    gs_path = f"{results_dir}/gs_{timestamp}"
    geo_wrapper.GSSavePointCloud(gs_path)
    console.print(f"[yellow][OUTPUT][/] Gaussian Splats: {gs_path}")

    hash_path = f"{results_dir}/hash_points_{timestamp}.ply"
    voxel_path = f"{results_dir}/voxel_points_{timestamp}.ply"
    geo_wrapper.serializeData(hash_path, voxel_path)
    console.print(f"[yellow][OUTPUT][/] Hash points: {hash_path}")
    console.print(f"[yellow][OUTPUT][/] Voxel points: {voxel_path}")

    geo_wrapper.clearBuffers()


def _create_geo_wrapper(data_cf, reader, results_dir):
    """
    Create and configure a GeoWrapper instance for Dynamic 3DGS mapping.

    Returns:
        (geo_wrapper, gs_visualize, end_frame)
    """
    from mrhash.src.pygeowrapper import GeoWrapper

    sdf_truncation = data_cf["map"]["sdf_truncation"]
    sdf_truncation_scale = data_cf["map"]["sdf_truncation_scale"]
    integration_weight_sample = data_cf["map"]["integration_weight_sample"]
    virtual_voxel_size = data_cf["map"]["virtual_voxel_size"]
    n_frames_invalidate_voxels = data_cf["map"]["n_frames_invalidate_voxels"]
    dynamic_detection = data_cf["map"].get("dynamic_detection", False)
    dynamic_method = data_cf["map"].get("dynamic_method", "tsdf_residual")
    # dynamic_detection is master switch; dynamic_method selects the approach
    # Enable C++ TSDF residual detection only for tsdf_residual method
    enable_tsdf_residual = dynamic_detection and dynamic_method == "tsdf_residual"
    dynamic_erosion_size = data_cf["map"].get("dynamic_erosion_size", 15)
    dynamic_dilation_size = data_cf["map"].get("dynamic_dilation_size", 10)
    dynamic_flood_threshold = data_cf["map"].get("dynamic_flood_threshold", 0.007)
    save_dynamic_mask = data_cf["map"].get("save_dynamic_mask", False)
    gs_only_dynamic_frames = data_cf["map"].get("gs_only_dynamic_frames", False)
    gs_visualize = data_cf["map"].get("gs_visualize", False)

    voxel_extents_scale = data_cf["streamer"]["voxel_extents_scale"]
    marching_cubes_threshold = data_cf["mesh"]["marching_cubes_threshold"]
    min_weight_threshold = data_cf["mesh"]["min_weight_threshold"]
    sdf_var_threshold = data_cf["mesh"]["sdf_var_threshold"]
    vertices_merging_threshold = data_cf["mesh"]["vertices_merging_threshold"]
    gs_optimization_param_path = data_cf.get("gs_optimization_param_path", "")

    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = data_cf["sensor"]["intrinsics"][0]
    K[1, 1] = data_cf["sensor"]["intrinsics"][1]
    K[0, 2] = data_cf["sensor"]["intrinsics"][2]
    K[1, 2] = data_cf["sensor"]["intrinsics"][3]
    K[2, 2] = 1

    img_rows = data_cf["sensor"]["resolution"][1]
    img_cols = data_cf["sensor"]["resolution"][0]
    min_depth = data_cf["sensor"]["min_depth"]
    max_depth = data_cf["sensor"]["max_depth"]

    end_frame = data_cf["end_frame"] if data_cf["end_frame"] != -1 else len(reader) + 1

    rgbd_camera = Camera(
        rows=img_rows, cols=img_cols, K=K,
        min_depth=min_depth, max_depth=max_depth,
        model=CameraModel.Pinhole,
    )

    geo_wrapper = GeoWrapper(
        sdf_truncation=sdf_truncation,
        sdf_truncation_scale=sdf_truncation_scale,
        integration_weight_sample=integration_weight_sample,
        virtual_voxel_size=virtual_voxel_size,
        n_frames_invalidate_voxels=n_frames_invalidate_voxels,
        voxel_extents_scale=voxel_extents_scale,
        viewer_active=False,
        marching_cubes_threshold=marching_cubes_threshold,
        min_weight_threshold=min_weight_threshold,
        sdf_var_threshold=sdf_var_threshold,
        gs_optimization_param_path=gs_optimization_param_path,
        vertices_merging_threshold=vertices_merging_threshold,
        projective_sdf=True,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    geo_wrapper.enableDynamicDetection(enable_tsdf_residual)
    geo_wrapper.setDynamicErosionSize(dynamic_erosion_size)
    geo_wrapper.setDynamicDilationSize(dynamic_dilation_size)
    geo_wrapper.setDynamicFloodThreshold(dynamic_flood_threshold)
    geo_wrapper.setGSOnlyDynamicFrames(gs_only_dynamic_frames)
    geo_wrapper.setGSVisualize(gs_visualize)

    if save_dynamic_mask:
        mask_dir = results_dir / "mrhash_mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        (mask_dir / "raw").mkdir(parents=True, exist_ok=True)
        geo_wrapper.setSaveDynamicMask(True)
        geo_wrapper.setMaskOutputPath(str(mask_dir))
        console.print(f"[green][3DGS-MAP][/] Saving masks to: {mask_dir}")

    geo_wrapper.setCamera(
        rgbd_camera.fx_, rgbd_camera.fy_,
        rgbd_camera.cx_, rgbd_camera.cy_,
        rgbd_camera.rows_, rgbd_camera.cols_,
        rgbd_camera.min_depth_, rgbd_camera.max_depth_,
        rgbd_camera.model_,
    )

    return geo_wrapper, gs_visualize, end_frame


def _map_frame(geo_wrapper, translation, quat, depth_img, rgb_img, gs_visualize, dynamic_mask=None):
    """Map a single frame through the 3DGS pipeline.
    
    Args:
        dynamic_mask: optional np.ndarray [H, W] uint8. 255 = dynamic, 0 = static.
    """
    geo_wrapper.setCurrPose(translation, quat)
    geo_wrapper.setDepthImage(depth_img)
    geo_wrapper.setRGBImage(rgb_img)
    if dynamic_mask is not None:
        geo_wrapper.setExternalDynamicMask(dynamic_mask)
    geo_wrapper.compute()

    if gs_visualize and geo_wrapper.hasGSRenderedImage():
        rendered = np.array(geo_wrapper.getGSRenderedImage())
        gt_display = rgb_img.astype(np.uint8)
        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        gt_bgr = cv2.cvtColor(gt_display, cv2.COLOR_RGB2BGR)
        combined = np.hstack([gt_bgr, rendered_bgr])
        cv2.imshow("3DGS: GT (left) | Rendered (right)", combined)
        cv2.waitKey(1)


def _save_results(geo_wrapper, results_dir, timestamp, gs_visualize):
    """Save mesh, Gaussian splats, and hash/voxel data."""
    if gs_visualize and geo_wrapper.hasGSRenderedImage():
        console.print(
            "[green][3DGS-MAP][/] Mapping complete. Press any key on the image window to close."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print_section("Saving Results", "yellow")
    geo_wrapper.streamAllOut()

    mesh_path = f"{results_dir}/mesh_{timestamp}.ply"
    geo_wrapper.extractMesh(mesh_path)
    console.print(f"[yellow][OUTPUT][/] Mesh: {mesh_path}")

    gs_path = f"{results_dir}/gs_{timestamp}"
    geo_wrapper.GSSavePointCloud(gs_path)
    console.print(f"[yellow][OUTPUT][/] Gaussian Splats: {gs_path}")

    hash_path = f"{results_dir}/hash_points_{timestamp}.ply"
    voxel_path = f"{results_dir}/voxel_points_{timestamp}.ply"
    geo_wrapper.serializeData(hash_path, voxel_path)
    console.print(f"[yellow][OUTPUT][/] Hash points: {hash_path}")
    console.print(f"[yellow][OUTPUT][/] Voxel points: {voxel_path}")

    geo_wrapper.clearBuffers()


def run_online_pipeline(data_cf, reader, timestamp):
    """
    Online mode: DROID-W tracking and Dynamic 3DGS mapping run simultaneously.

    For each frame: DROID-W estimates the pose → immediately feed it to 3DGS mapping.
    During warmup, frames are buffered and mapped retroactively once poses are available.

    Args:
        data_cf: full config dict
        reader: TUMUnifiedReader instance
        timestamp: string timestamp for output files
    """
    print_section("Online Mode: Track + Map Simultaneously", "magenta")

    # --- Setup DROID-W online tracker ---
    import torch
    import droidw
    from droidw.config_adapter import build_droidw_config
    from droidw.dataset_wrapper import DroidWTUMDataset
    from droidw.slam_wrapper import OnlineDroidWTracker

    droidw_cfg = build_droidw_config(data_cf)

    color_paths = reader.get_color_paths()
    depth_paths = reader.get_depth_paths()
    gt_poses = reader.get_gt_poses_4x4()

    droidw_dataset = DroidWTUMDataset(
        cfg=droidw_cfg,
        color_paths=color_paths,
        depth_paths=depth_paths,
        gt_poses_4x4=gt_poses,
        device=droidw_cfg["device"],
    )

    # Save GT poses if available
    if gt_poses is not None and droidw_cfg.get("save_gt_poses", True):
        output_dir = Path(data_cf["results_path"]) / "droidw"
        output_dir.mkdir(parents=True, exist_ok=True)
        droidw_dataset.save_gt_poses(str(output_dir / "gt_poses.txt"), gt_poses)

    tracker = OnlineDroidWTracker(droidw_cfg, droidw_dataset)

    # --- Setup Dynamic 3DGS mapper ---
    results_dir = Path(data_cf["results_path"])
    geo_wrapper, gs_visualize, end_frame = _create_geo_wrapper(data_cf, reader, results_dir)

    # --- Print config info ---
    dynamic_detection = data_cf["map"].get("dynamic_detection", False)
    dynamic_method = data_cf["map"].get("dynamic_method", "tsdf_residual")
    uncertainty_threshold = data_cf["map"].get("droidw_uncertainty_threshold", 0.9)
    uncertainty_dilation = data_cf["map"].get("droidw_uncertainty_dilation", 10)
    use_droidw_uncertainty = (dynamic_detection and dynamic_method == "droidw_uncertainty")
    map_keyframes_only = data_cf["map"].get("map_keyframes_only", False)

    table = Table(title="Online Pipeline Parameters", box=box.SIMPLE)
    table.add_column("Parameter", style="magenta")
    table.add_column("Value", style="white")
    table.add_row("Mode", "Online (Track + Map)")
    table.add_row("Total Frames", str(len(reader)))
    table.add_row("DROID-W Warmup", str(droidw_cfg["tracking"]["warmup"]))
    table.add_row("Dynamic Detection", str(dynamic_detection))
    table.add_row("Dynamic Method", dynamic_method)
    if use_droidw_uncertainty:
        table.add_row("Uncertainty Threshold", str(uncertainty_threshold))
        table.add_row("Uncertainty Dilation", str(uncertainty_dilation))
    table.add_row("Map Keyframes Only", str(map_keyframes_only))
    table.add_row("GS Visualize", str(gs_visualize))
    console.print(table)

    img_h = data_cf["sensor"]["resolution"][1]
    img_w = data_cf["sensor"]["resolution"][0]

    # --- Main online loop ---
    num_frames = len(reader)
    mapped_frames = 0
    warmup_frame_count = 0

    console.print(
        f"[magenta][ONLINE][/] Starting online pipeline on {num_frames} frames..."
    )

    for idx in tqdm(range(num_frames), desc="[ONLINE] Track+Map"):
        frame, _, _, depth_img, rgb_img = reader[idx]
        if frame > end_frame:
            break

        # --- Step 1: DROID-W tracking ---
        result = tracker.process_frame(idx)

        # Free intermediate tracking GPU memory before mapping
        torch.cuda.empty_cache()

        # --- Step 2: 3DGS mapping ---
        if result["just_initialized"]:
            # Warmup just completed: retroactively map all frames [0, idx]
            # using interpolated poses from warmup keyframes
            console.print(
                f"\n[magenta][ONLINE][/] Warmup done at frame {idx}. "
                f"Retroactively mapping frames 0-{idx}..."
            )

            # Build interpolated poses for warmup frames
            warmup_poses = result["warmup_poses"]  # (frame_idx, t, q) for keyframes
            kf_indices = np.array([p[0] for p in warmup_poses])
            kf_translations = np.array([p[1] for p in warmup_poses])
            kf_quaternions = np.array([p[2] for p in warmup_poses])

            # Determine which frames to map during warmup
            if map_keyframes_only:
                warmup_map_indices = sorted(int(kf_indices[i]) for i in range(len(kf_indices)))
            else:
                warmup_map_indices = list(range(idx + 1))

            for warmup_idx in warmup_map_indices:
                w_frame, _, _, w_depth, w_rgb = reader[warmup_idx]
                if w_frame > end_frame:
                    break
                # Find nearest keyframe pose for this frame
                diffs = np.abs(kf_indices - warmup_idx)
                nearest = np.argmin(diffs)
                t = kf_translations[nearest]
                q = kf_quaternions[nearest]
                # Get uncertainty mask if using droidw_uncertainty method
                mask = None
                if use_droidw_uncertainty:
                    mask = tracker.get_uncertainty_mask(warmup_idx, img_h, img_w, uncertainty_threshold, uncertainty_dilation)
                _map_frame(geo_wrapper, t, q, w_depth, w_rgb, gs_visualize, mask)
                mapped_frames += 1

        elif result["pose_available"]:
            # Skip non-keyframes when map_keyframes_only is enabled
            if map_keyframes_only and not result["is_keyframe"]:
                # Still visualize: render from current pose without mapping
                if gs_visualize:
                    geo_wrapper.setCurrPose(result["translation"], result["quaternion"])
                    geo_wrapper.GSRenderOnly()
                    gt_bgr = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    if geo_wrapper.hasGSRenderedImage():
                        rendered = np.array(geo_wrapper.getGSRenderedImage())
                        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                        combined = np.hstack([gt_bgr, rendered_bgr])
                    else:
                        combined = gt_bgr
                    cv2.imshow("3DGS: GT (left) | Rendered (right)", combined)
                    cv2.waitKey(1)
                continue
            # Normal online frame: track + map immediately
            mask = None
            if use_droidw_uncertainty:
                mask = tracker.get_uncertainty_mask(idx, img_h, img_w, uncertainty_threshold, uncertainty_dilation)
            _map_frame(
                geo_wrapper,
                result["translation"],
                result["quaternion"],
                depth_img,
                rgb_img,
                gs_visualize,
                mask,
            )
            mapped_frames += 1
        else:
            warmup_frame_count += 1

    console.print(
        f"\n[magenta][ONLINE][/] Pipeline complete: "
        f"{mapped_frames} frames mapped, "
        f"{warmup_frame_count} frames in warmup phase"
    )

    # --- Save DROID-W estimated poses ---
    output_dir = Path(data_cf["results_path"]) / "droidw"
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_data = tracker.get_all_poses()
    np.savez(
        str(output_dir / "estimated_poses.npz"),
        keyframe_indices=pose_data["keyframe_indices"],
        all_translations=pose_data["all_translations"],
        all_quaternions=pose_data["all_quaternions"],
        all_poses_4x4=pose_data["all_poses_4x4"],
    )
    console.print(
        f"[blue][DROID-W][/] Estimated poses saved to {output_dir / 'estimated_poses.npz'}"
    )

    # --- Finalize tracker ---
    tracker.finalize()

    # --- Save 3DGS results ---
    _save_results(geo_wrapper, results_dir, timestamp, gs_visualize)


def main(
    config_path: Annotated[
        str, typer.Argument(help="Path to the integrated config file")
    ] = "mrhash/configurations/tum_integrated.cfg",
) -> None:
    """DROID-W + Dynamic 3DGS Integrated Pipeline for TUM RGB-D datasets."""

    # Must set spawn before ANY multiprocessing objects are created.
    # This is required because DROID-W's DepthVideo and other objects create
    # mp.Value / mp.Lock with the default context.  All mp primitives must
    # share the same context.
    # The C++ pygeowrapper binding now releases the GIL before calling
    # backward(), so Phase 2 works correctly with spawn set.
    import torch.multiprocessing as torch_mp
    try:
        torch_mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    print_banner()

    # --- Load Config ---
    config = Path(config_path)
    if not config.exists():
        console.print(f"[red]Error: Config file {config} does not exist!")
        sys.exit(1)

    with open(config, "r") as f:
        data_cf = yaml.safe_load(f)

    data_path = Path(data_cf["data_path"])
    results_dir = Path(data_cf["results_path"])
    use_gt_pose = data_cf.get("use_gt_pose", False)

    if not data_path.exists():
        console.print(f"[red]Error: Data path {data_path} does not exist!")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    copied_config = results_dir / f"{timestamp}_{config.name}"
    shutil.copy(config, copied_config)

    # Print config summary
    print_config_summary(data_cf, use_gt_pose)

    # --- Load Data ---
    print_section("Loading TUM RGB-D Dataset", "yellow")

    sensor = data_cf["sensor"]
    reader = TUMUnifiedReader(
        data_dir=data_path,
        min_range=sensor["min_depth"],
        max_range=sensor["max_depth"],
        depth_scaling=sensor["depth_scaling"],
        load_gt_pose=True,  # Always load GT for evaluation, even if not used for mapping
    )

    console.print(f"[yellow][DATA][/] Loaded {len(reader)} frames from {data_path}")
    console.print(f"[yellow][DATA][/] GT poses available: {reader.has_gt_pose}")

    # --- Check mode ---
    online_mode = data_cf.get("online_mode", False)

    if online_mode and not use_gt_pose:
        # === Online Mode: Track + Map simultaneously ===
        # Enable expandable_segments to reduce CUDA fragmentation in online mode.
        # NOTE: incompatible with CUDA IPC (multiprocessing), so only set for online mode.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        console.print(
            "\n[bold magenta]Mode: ONLINE (Track + Map simultaneously)[/]\n"
        )
        run_online_pipeline(data_cf, reader, timestamp)
    else:
        if online_mode and use_gt_pose:
            console.print(
                "\n[yellow]Warning: online_mode ignored when use_gt_pose=true "
                "(no tracking needed)[/]\n"
            )

        # === Offline Mode (original): Phase 1 then Phase 2 ===
        pose_data = None
        slam_wrapper = None
        if not use_gt_pose:
            pose_data, slam_wrapper = run_droidw_tracking(data_cf, reader)
        else:
            console.print(
                "\n[green]Skipping DROID-W tracking (use_gt_pose=true)[/]\n"
            )

        # Cleanup DROID-W GPU resources before Phase 2
        # Keep slam_wrapper alive if we need uncertainty masks
        dynamic_method = data_cf["map"].get("dynamic_method", "none")
        import gc
        import torch
        torch.cuda.empty_cache()
        gc.collect()

        # Phase 2: Dynamic 3DGS Mapping
        run_dynamic_3dgs_mapping(data_cf, reader, pose_data, use_gt_pose, timestamp, slam_wrapper)

    # --- Done ---
    console.print(
        f"\n[bold green]{'═' * 60}[/]"
    )
    console.print(
        f"[bold green]  Pipeline Complete![/]"
    )
    console.print(
        f"[bold green]  Results saved to: {results_dir}[/]"
    )
    console.print(
        f"[bold green]{'═' * 60}[/]\n"
    )


def run():
    typer.run(main)


if __name__ == "__main__":
    run()
