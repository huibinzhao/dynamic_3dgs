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
import csv
import math
from collections import defaultdict
from contextlib import contextmanager, nullcontext
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

_mpl_config_dir = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "dynamic_3dgs_mpl"
)
os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir)
try:
    os.makedirs(_mpl_config_dir, exist_ok=True)
except OSError:
    pass

import cv2
import numpy as np
import typer
import yaml
from scipy.spatial.transform import Rotation, Slerp
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


def _format_float(value):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def _sync_cuda_if_available():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class TimerSummary:
    def __init__(self):
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._total_start = time.perf_counter()

    @contextmanager
    def measure(self, name, sync_cuda=False):
        if sync_cuda:
            _sync_cuda_if_available()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                _sync_cuda_if_available()
            elapsed = time.perf_counter() - start
            self.add(name, elapsed)

    def add(self, name, elapsed_seconds):
        self._totals[name] += float(elapsed_seconds)
        self._counts[name] += 1

    def write(self, output_path):
        total_elapsed = time.perf_counter() - self._total_start
        self._totals["total_pipeline"] = total_elapsed
        self._counts["total_pipeline"] = 1
        rows = []
        for name in sorted(self._totals):
            total_s = self._totals[name]
            count = self._counts[name]
            avg_ms = (total_s / count * 1000.0) if count else 0.0
            percent = (total_s / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            rows.append((name, count, total_s, avg_ms, percent))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("module,count,total_seconds,average_ms,percent_of_pipeline\n")
            for name, count, total_s, avg_ms, percent in rows:
                f.write(
                    f"{name},{count},{total_s:.6f},{avg_ms:.3f},{percent:.2f}\n"
                )


class GSMetricRecorder:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.rendered_dir = self.results_dir / "3dgs_rendered"
        self.frames = []
        self.rows = []
        self.lpips_model = None
        self.lpips_device = None
        self.lpips_status = "not_initialized"

    def add_training_frame(self, translation, quat, frame_idx, frame_number, mapped_frame_id, gt_rgb):
        self.frames.append({
            "translation": np.asarray(translation, dtype=np.float32).copy(),
            "quat": np.asarray(quat, dtype=np.float32).copy(),
            "frame_idx": int(frame_idx),
            "frame_number": int(frame_number),
            "mapped_frame_id": int(mapped_frame_id),
            "gt_rgb": np.asarray(gt_rgb, dtype=np.uint8).copy(),
        })

    def render_final_metrics(self, geo_wrapper):
        self.rows = []
        for frame in tqdm(self.frames, desc="[3DGS-METRICS] Final render"):
            geo_wrapper.setCurrPose(frame["translation"], frame["quat"])
            geo_wrapper.GSRenderOnly()
            if not geo_wrapper.hasGSRenderedImage():
                self.rows.append({
                    "frame_idx": frame["frame_idx"],
                    "frame_number": frame["frame_number"],
                    "mapped_frame_id": frame["mapped_frame_id"],
                    "psnr": float("nan"),
                    "ssim": float("nan"),
                    "lpips": float("nan"),
                })
                continue

            rendered = np.array(geo_wrapper.getGSRenderedImage())
            rendered, gt = self._align_images(rendered, frame["gt_rgb"])
            self._save_rendered_image(rendered, frame["frame_idx"], frame["mapped_frame_id"])
            self.rows.append({
                "frame_idx": frame["frame_idx"],
                "frame_number": frame["frame_number"],
                "mapped_frame_id": frame["mapped_frame_id"],
                "psnr": self._psnr(rendered, gt),
                "ssim": self._ssim(rendered, gt),
                "lpips": self._lpips(rendered, gt),
            })

    def write(self, output_path=None):
        output_path = Path(output_path) if output_path is not None else self.results_dir / "3dgs_frame_metrics.csv"
        fieldnames = ["frame_idx", "frame_number", "mapped_frame_id", "psnr", "ssim", "lpips"]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: _format_float(row[key]) if key in ("psnr", "ssim", "lpips") else row[key] for key in fieldnames})
            averages = self.averages()
            writer.writerow({
                "frame_idx": "average",
                "frame_number": "",
                "mapped_frame_id": len(self.rows),
                "psnr": _format_float(averages["psnr"]),
                "ssim": _format_float(averages["ssim"]),
                "lpips": _format_float(averages["lpips"]),
            })
        return output_path

    def _save_rendered_image(self, rendered, frame_idx, mapped_frame_id):
        self.rendered_dir.mkdir(parents=True, exist_ok=True)
        rendered_uint8 = np.clip(rendered, 0, 255).astype(np.uint8)
        rendered_bgr = cv2.cvtColor(rendered_uint8, cv2.COLOR_RGB2BGR)
        output_path = self.rendered_dir / f"rendered_{int(mapped_frame_id):06d}_frame_{int(frame_idx):06d}.png"
        cv2.imwrite(str(output_path), rendered_bgr)

    def averages(self):
        return {
            key: self._mean_finite([row[key] for row in self.rows])
            for key in ("psnr", "ssim", "lpips")
        }

    @staticmethod
    def _align_images(rendered, gt):
        rendered = np.asarray(rendered, dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)
        if rendered.ndim == 2:
            rendered = cv2.cvtColor(rendered, cv2.COLOR_GRAY2RGB)
        if gt.ndim == 2:
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
        if rendered.shape[:2] != gt.shape[:2]:
            rendered = cv2.resize(rendered, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        return rendered[..., :3], gt[..., :3]

    @staticmethod
    def _psnr(rendered, gt):
        rendered_f = rendered.astype(np.float32) / 255.0
        gt_f = gt.astype(np.float32) / 255.0
        mse = np.mean((rendered_f - gt_f) ** 2)
        if mse <= 1e-12:
            return float("inf")
        return float(20.0 * np.log10(1.0 / np.sqrt(mse)))

    @staticmethod
    def _ssim(rendered, gt):
        img1 = rendered.astype(np.float32) / 255.0
        img2 = gt.astype(np.float32) / 255.0
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        kernel = (11, 11)
        sigma = 1.5
        mu1 = cv2.GaussianBlur(img1, kernel, sigma)
        mu2 = cv2.GaussianBlur(img2, kernel, sigma)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel, sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu1_mu2
        ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return float(np.mean(ssim_map))

    def _lpips(self, rendered, gt):
        if self.lpips_status == "unavailable":
            return float("nan")
        try:
            import torch
            import lpips

            if self.lpips_model is None:
                self.lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.lpips_model = lpips.LPIPS(net="alex").to(self.lpips_device).eval()
                self.lpips_status = "available"

            rendered_tensor = self._lpips_tensor(rendered, torch, self.lpips_device)
            gt_tensor = self._lpips_tensor(gt, torch, self.lpips_device)
            with torch.no_grad():
                return float(self.lpips_model(rendered_tensor, gt_tensor).item())
        except Exception as exc:
            self.lpips_status = "unavailable"
            console.print(f"[yellow][METRICS][/] LPIPS unavailable: {exc}. Writing nan for LPIPS.")
            return float("nan")

    @staticmethod
    def _lpips_tensor(image, torch_module, device):
        tensor = torch_module.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        return tensor.to(device)

    @staticmethod
    def _mean_finite(values):
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return float("nan")
        return float(np.mean(finite))


def _write_tum_pose_file(output_path, poses_4x4, timestamps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for timestamp, pose in zip(timestamps, poses_4x4):
            pose = np.asarray(pose, dtype=np.float64)
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            t = pose[:3, 3]
            f.write(
                f"{float(timestamp):.6f} "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )


def _valid_trajectory_rows(est_poses, ref_poses, timestamps):
    n = min(len(est_poses), len(ref_poses), len(timestamps))
    est_poses = np.asarray(est_poses[:n], dtype=np.float64)
    ref_poses = np.asarray(ref_poses[:n], dtype=np.float64)
    timestamps = np.asarray(timestamps[:n], dtype=np.float64)

    valid = (
        np.isfinite(est_poses).all(axis=(1, 2))
        & np.isfinite(ref_poses).all(axis=(1, 2))
        & np.isfinite(timestamps)
    )
    return est_poses[valid], ref_poses[valid], timestamps[valid]


def _ape_statistics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "min": float(np.min(errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "sse": float(np.sum(errors * errors)),
        "std": float(np.std(errors)),
    }


def _umeyama_align_positions(est_xyz, ref_xyz):
    est_xyz = np.asarray(est_xyz, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    mu_est = np.mean(est_xyz, axis=0)
    mu_ref = np.mean(ref_xyz, axis=0)
    est_centered = est_xyz - mu_est
    ref_centered = ref_xyz - mu_ref

    cov = (ref_centered.T @ est_centered) / len(est_xyz)
    u, singular_values, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    variance = float(np.sum(est_centered * est_centered) / len(est_xyz))
    if variance <= 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = mu_ref - scale * rotation @ mu_est
    aligned_xyz = scale * (rotation @ est_xyz.T).T + translation
    return aligned_xyz, rotation, translation, scale


def _ensure_matplotlib_config_dir():
    config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "dynamic_3dgs_mpl"
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    config_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def _temporary_writable_evo_home():
    if "evo.tools.settings" in sys.modules:
        yield
        return

    home = Path.home()
    evo_dir = home / ".evo"
    writable_target = evo_dir if evo_dir.exists() else home
    if os.access(writable_target, os.W_OK):
        yield
        return

    tmp_home = Path(os.environ.get("TMPDIR", "/tmp")) / "dynamic_3dgs_evo_home"
    tmp_home.mkdir(parents=True, exist_ok=True)
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp_home)
    try:
        yield
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home


def _plot_trajectory_xy(output_path, est_xyz, ref_xyz, errors=None, title=None):
    try:
        _ensure_matplotlib_config_dir()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], "--", color="0.45", label="reference")
        if errors is None:
            ax.plot(est_xyz[:, 0], est_xyz[:, 1], color="tab:blue", label="estimated")
        else:
            sc = ax.scatter(
                est_xyz[:, 0],
                est_xyz[:, 1],
                c=errors,
                cmap="viridis",
                s=8,
                label="estimated",
            )
            fig.colorbar(sc, ax=ax, label="APE translation error (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True)
        ax.legend()
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path
    except Exception as exc:
        console.print(f"[yellow][EVAL][/] Trajectory plot skipped: {exc}")
        return None


def _evaluate_with_evo(est_poses, ref_poses, timestamps, plot_path):
    _ensure_matplotlib_config_dir()
    with _temporary_writable_evo_home():
        from evo.core import metrics, sync
        from evo.core.trajectory import PoseTrajectory3D

        traj_est = PoseTrajectory3D(poses_se3=list(est_poses), timestamps=timestamps)
        traj_ref = PoseTrajectory3D(poses_se3=list(ref_poses), timestamps=timestamps)
        traj_ref, traj_est = sync.associate_trajectories(
            traj_ref, traj_est, max_diff=0.1
        )
        rotation, translation, scale = traj_est.align(traj_ref, correct_scale=True)

        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_ref, traj_est))
        stats = {
            key: float(value)
            for key, value in ape_metric.get_all_statistics().items()
        }

        _plot_trajectory_xy(
            plot_path,
            traj_est.positions_xyz,
            traj_ref.positions_xyz,
            ape_metric.error,
            title="APE mapped onto trajectory, RMSE: %.3f cm"
            % (stats["rmse"] * 100.0),
        )

        return stats, rotation, translation, scale, len(traj_est.positions_xyz), "evo"


def _evaluate_with_numpy(est_poses, ref_poses, plot_path):
    est_xyz = est_poses[:, :3, 3]
    ref_xyz = ref_poses[:, :3, 3]
    aligned_xyz, rotation, translation, scale = _umeyama_align_positions(
        est_xyz, ref_xyz
    )
    errors = np.linalg.norm(aligned_xyz - ref_xyz, axis=1)
    stats = _ape_statistics(errors)
    _plot_trajectory_xy(
        plot_path,
        aligned_xyz,
        ref_xyz,
        errors,
        title="APE mapped onto trajectory, RMSE: %.3f cm"
        % (stats["rmse"] * 100.0),
    )
    return stats, rotation, translation, scale, len(est_xyz), "numpy_umeyama"


def _write_localization_status(metrics_path, status, message):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("##########Full traj##########\n")
        f.write(f"status: {status}\n")
        f.write(f"message: {message}\n")
    return metrics_path


def write_localization_accuracy(data_cf, reader, pose_data):
    results_dir = Path(data_cf["results_path"])
    traj_dir = results_dir / "traj"
    metrics_path = traj_dir / "metrics_full_traj.txt"
    plot_path = traj_dir / "full_traj.png"

    if pose_data is None or "all_poses_4x4" not in pose_data:
        return _write_localization_status(
            metrics_path,
            "skipped",
            "No estimated DROID-W trajectory is available.",
        )

    ref_poses = reader.get_gt_poses_4x4()
    if ref_poses is None:
        return _write_localization_status(
            metrics_path,
            "skipped",
            "No ground-truth trajectory is available for localization evaluation.",
        )

    est_poses = np.asarray(pose_data["all_poses_4x4"], dtype=np.float64)
    ref_poses = np.asarray(ref_poses, dtype=np.float64)
    timestamps = np.asarray(reader.get_timestamps(), dtype=np.float64)
    if est_poses.ndim != 3 or est_poses.shape[1:] != (4, 4):
        return _write_localization_status(
            metrics_path,
            "failed",
            f"Invalid estimated pose array shape: {est_poses.shape}.",
        )

    est_poses, ref_poses, timestamps = _valid_trajectory_rows(
        est_poses, ref_poses, timestamps
    )
    if len(est_poses) < 2:
        return _write_localization_status(
            metrics_path,
            "skipped",
            "Need at least two valid estimated/ground-truth poses.",
        )

    _write_tum_pose_file(traj_dir / "est_poses_full.txt", est_poses, timestamps)
    _write_tum_pose_file(traj_dir / "gt_poses_full.txt", ref_poses, timestamps)

    try:
        stats, rotation, translation, scale, num_pairs, method = _evaluate_with_evo(
            est_poses, ref_poses, timestamps, plot_path
        )
    except Exception as exc:
        console.print(
            f"[yellow][EVAL][/] evo localization evaluation unavailable: {exc}. "
            "Falling back to NumPy Umeyama alignment."
        )
        stats, rotation, translation, scale, num_pairs, method = _evaluate_with_numpy(
            est_poses, ref_poses, plot_path
        )

    output_str = "#" * 10 + "Full traj" + "#" * 10 + "\n"
    output_str += f"scale: {scale}\n"
    output_str += f"rotation:\n{rotation}\n"
    output_str += f"translation:{translation}\n"
    output_str += f"statistics:\n{stats}"
    output_str += f"\nmethod: {method}"
    output_str += f"\nnum_pairs: {num_pairs}"
    output_str += f"\nrmse_cm: {stats['rmse'] * 100.0:.6f}\n"

    traj_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(output_str)

    console.print(
        f"[yellow][OUTPUT][/] Localization ATE metrics: {metrics_path} "
        f"(RMSE {stats['rmse'] * 100.0:.3f} cm)"
    )
    return metrics_path


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
    table.add_row("3DGS Map", "Evaluate Metrics", str(m.get("evaluate_3dgs_metrics", True)))
    table.add_row("3DGS Map", "Keyframe Delay", str(m.get("mapping_keyframe_delay", 5)))

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


def run_droidw_tracking(data_cf, reader, timer_summary=None):
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
    if timer_summary is not None:
        with timer_summary.measure("droidw_tracking_total", sync_cuda=True):
            pose_data = slam_wrapper.run_tracking()
    else:
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

    eval_timer = timer_summary.measure("localization_accuracy_eval", sync_cuda=False) if timer_summary is not None else nullcontext()
    with eval_timer:
        write_localization_accuracy(data_cf, reader, pose_data)

    return pose_data, slam_wrapper


def run_dynamic_3dgs_mapping(data_cf, reader, pose_data, use_gt_pose, timestamp, slam_wrapper=None, timer_summary=None):
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
    evaluate_3dgs_metrics = data_cf["map"].get("evaluate_3dgs_metrics", True)
    metric_recorder = GSMetricRecorder(results_dir) if evaluate_3dgs_metrics else None

    # Load map config
    sdf_truncation = data_cf["map"]["sdf_truncation"]
    sdf_truncation_scale = data_cf["map"]["sdf_truncation_scale"]
    integration_weight_sample = data_cf["map"]["integration_weight_sample"]
    virtual_voxel_size = data_cf["map"]["virtual_voxel_size"]
    n_frames_invalidate_voxels = data_cf["map"]["n_frames_invalidate_voxels"]
    dynamic_detection = data_cf["map"].get("dynamic_detection", False)
    dynamic_method = data_cf["map"].get("dynamic_method", "tsdf_residual")
    # dynamic_detection is master switch; dynamic_method selects the approach
    # Enable C++ TSDF residual detection only when it is the standalone method.
    # The fused method computes TSDF residuals inside the fusion branch.
    enable_tsdf_residual = dynamic_detection and dynamic_method == "tsdf_residual"
    enable_dynamic_fusion = dynamic_detection and dynamic_method == "fused_droidw_tsdf"
    dynamic_erosion_size = data_cf["map"].get("dynamic_erosion_size", 15)
    dynamic_dilation_size = data_cf["map"].get("dynamic_dilation_size", 10)
    dynamic_flood_threshold = data_cf["map"].get("dynamic_flood_threshold", 0.007)
    save_keyframe_plots = data_cf["map"].get("save_keyframe_comparison_plots", False)
    save_dynamic_mask = data_cf["map"].get("save_dynamic_mask", False) or save_keyframe_plots
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
    use_droidw_uncertainty = (
        dynamic_detection
        and dynamic_method in ("droidw_uncertainty", "fused_droidw_tsdf")
        and slam_wrapper is not None
    )
    uncertainty_threshold = data_cf["map"].get("droidw_uncertainty_threshold", 0.9)
    uncertainty_dilation = data_cf["map"].get("droidw_uncertainty_dilation", 10)
    map_keyframes_only = data_cf["map"].get("map_keyframes_only", False)
    mapping_keyframe_delay = _get_mapping_keyframe_delay(data_cf)
    delayed_mapping = (
        _delayed_keyframe_mapping_enabled(data_cf, use_gt_pose)
        and pose_data is not None
        and "keyframe_indices" in pose_data
    )

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
    table.add_row("Evaluate 3DGS Metrics", str(evaluate_3dgs_metrics))
    table.add_row("Keyframe Comparison Plots", str(save_keyframe_plots))
    table.add_row("Keyframe Mapping Delay", str(mapping_keyframe_delay if delayed_mapping else 0))
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
    geo_wrapper.enableDynamicFusion(enable_dynamic_fusion)
    geo_wrapper.setDynamicFusionWeights(
        float(data_cf["map"].get("fused_uncertainty_weight", 0.6)),
        float(data_cf["map"].get("fused_tsdf_weight", 0.4)),
    )
    geo_wrapper.setDynamicFusionThreshold(float(data_cf["map"].get("fused_dynamic_threshold", 0.5)))
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
        with open(mask_dir / "mask_index_map.txt", "w", encoding="utf-8") as f:
            f.write("mask_id,frame_idx,frame_number,affine_source_frame_idx\n")
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

    geo_wrapper_frame_count = 0
    mask_index_file = results_dir / "mrhash_mask" / "mask_index_map.txt" if save_dynamic_mask else None
    rgb_output_dir = results_dir / "rgb"
    rgb_output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green][3DGS-MAP][/] Saving keyframe RGB images to: {rgb_output_dir}")
    plot_output_dir = _get_keyframe_plot_dir(data_cf, results_dir) if save_keyframe_plots else None
    if plot_output_dir is not None:
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green][3DGS-MAP][/] Saving keyframe comparison plots to: {plot_output_dir}")
    plot_keyframe_set = _get_pose_keyframe_set(pose_data)
    comparison_kf_count = 0

    if delayed_mapping:
        keyframe_indices = [int(k) for k in pose_data["keyframe_indices"]]
        warmup_kf_count = _get_warmup_keyframe_count(data_cf)
        source_start_pos = max(0, warmup_kf_count + mapping_keyframe_delay - 1)
        console.print(
            "[green][3DGS-MAP][/] Delayed keyframe mapping: "
            f"delay={mapping_keyframe_delay}, start trigger keyframe #{source_start_pos + 1}"
        )

        last_mapped_target_pos = -1
        for source_pos in tqdm(
            range(source_start_pos, len(keyframe_indices)),
            desc="[3DGS-MAP] Delayed Mapping",
        ):
            affine_source_idx = keyframe_indices[source_pos]
            if affine_source_idx > end_frame:
                break

            eligible_target_pos = source_pos - mapping_keyframe_delay
            while last_mapped_target_pos < eligible_target_pos:
                target_pos = last_mapped_target_pos + 1
                target_idx = keyframe_indices[target_pos]
                if target_idx >= len(reader) or affine_source_idx >= len(reader):
                    last_mapped_target_pos = target_pos
                    continue

                frame, _, _, depth_img, rgb_img = reader[target_idx]
                if frame > end_frame:
                    break

                translation = pose_data["all_translations"][target_idx]
                quat = pose_data["all_quaternions"][target_idx]
                comparison_kf_count = _map_frame_with_outputs(
                    geo_wrapper,
                    translation,
                    quat,
                    depth_img,
                    rgb_img,
                    target_idx,
                    frame,
                    geo_wrapper_frame_count,
                    gs_visualize,
                    dynamic_method,
                    use_droidw_uncertainty,
                    slam_wrapper,
                    img_rows,
                    img_cols,
                    uncertainty_threshold,
                    uncertainty_dilation,
                    results_dir,
                    rgb_output_dir,
                    mask_index_file,
                    plot_output_dir,
                    comparison_kf_count,
                    affine_source_frame_idx=affine_source_idx,
                    save_plot=True,
                    metric_recorder=metric_recorder,
                    timer_summary=timer_summary,
                )
                geo_wrapper_frame_count += 1
                last_mapped_target_pos = target_pos

        if keyframe_indices and last_mapped_target_pos < len(keyframe_indices) - 1:
            affine_source_idx = keyframe_indices[-1]
            console.print(
                "[green][3DGS-MAP][/] Flushing delayed tail keyframes without "
                f"delay constraint: #{last_mapped_target_pos + 2}-#{len(keyframe_indices)}"
            )
            while last_mapped_target_pos < len(keyframe_indices) - 1:
                target_pos = last_mapped_target_pos + 1
                target_idx = keyframe_indices[target_pos]
                if target_idx >= len(reader) or affine_source_idx >= len(reader):
                    last_mapped_target_pos = target_pos
                    continue

                frame, _, _, depth_img, rgb_img = reader[target_idx]
                if frame > end_frame:
                    break

                translation = pose_data["all_translations"][target_idx]
                quat = pose_data["all_quaternions"][target_idx]
                comparison_kf_count = _map_frame_with_outputs(
                    geo_wrapper,
                    translation,
                    quat,
                    depth_img,
                    rgb_img,
                    target_idx,
                    frame,
                    geo_wrapper_frame_count,
                    gs_visualize,
                    dynamic_method,
                    use_droidw_uncertainty,
                    slam_wrapper,
                    img_rows,
                    img_cols,
                    uncertainty_threshold,
                    uncertainty_dilation,
                    results_dir,
                    rgb_output_dir,
                    mask_index_file,
                    plot_output_dir,
                    comparison_kf_count,
                    affine_source_frame_idx=affine_source_idx,
                    save_plot=True,
                    metric_recorder=metric_recorder,
                    timer_summary=timer_summary,
                )
                geo_wrapper_frame_count += 1
                last_mapped_target_pos = target_pos
    else:
        for idx in tqdm(range(len(reader)), desc="[3DGS-MAP] Mapping"):
            # Skip non-keyframes when map_keyframes_only is enabled
            if keyframe_set is not None and idx not in keyframe_set:
                # Still visualize all frames in the non-delayed legacy path.
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

            save_plot = _should_save_keyframe_plot(idx, plot_keyframe_set)

            comparison_kf_count = _map_frame_with_outputs(
                geo_wrapper,
                translation,
                quat,
                depth_img,
                rgb_img,
                idx,
                frame,
                geo_wrapper_frame_count,
                gs_visualize,
                dynamic_method,
                use_droidw_uncertainty,
                slam_wrapper,
                img_rows,
                img_cols,
                uncertainty_threshold,
                uncertainty_dilation,
                results_dir,
                rgb_output_dir,
                mask_index_file,
                plot_output_dir,
                comparison_kf_count,
                save_plot=save_plot,
                metric_recorder=metric_recorder,
                timer_summary=timer_summary,
            )
            geo_wrapper_frame_count += 1

    if gs_visualize and geo_wrapper.hasGSRenderedImage():
        console.print(
            "[green][3DGS-MAP][/] Mapping complete. Press any key on the image window to close."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save outputs
    print_section("Saving Results", "yellow")

    if timer_summary is not None:
        save_timer = timer_summary.measure("save_3dgs_results", sync_cuda=True)
    else:
        save_timer = nullcontext()

    with save_timer:
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

    if metric_recorder is not None:
        metric_timer = timer_summary.measure("3dgs_final_metric_render", sync_cuda=True) if timer_summary is not None else nullcontext()
        with metric_timer:
            metric_recorder.render_final_metrics(geo_wrapper)
        metrics_path = metric_recorder.write()
        console.print(f"[yellow][OUTPUT][/] 3DGS frame metrics: {metrics_path}")

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
    # Enable C++ TSDF residual detection only when it is the standalone method.
    # The fused method computes TSDF residuals inside the fusion branch.
    enable_tsdf_residual = dynamic_detection and dynamic_method == "tsdf_residual"
    enable_dynamic_fusion = dynamic_detection and dynamic_method == "fused_droidw_tsdf"
    dynamic_erosion_size = data_cf["map"].get("dynamic_erosion_size", 15)
    dynamic_dilation_size = data_cf["map"].get("dynamic_dilation_size", 10)
    dynamic_flood_threshold = data_cf["map"].get("dynamic_flood_threshold", 0.007)
    save_keyframe_plots = data_cf["map"].get("save_keyframe_comparison_plots", False)
    save_dynamic_mask = data_cf["map"].get("save_dynamic_mask", False) or save_keyframe_plots
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
    geo_wrapper.enableDynamicFusion(enable_dynamic_fusion)
    geo_wrapper.setDynamicFusionWeights(
        float(data_cf["map"].get("fused_uncertainty_weight", 0.6)),
        float(data_cf["map"].get("fused_tsdf_weight", 0.4)),
    )
    geo_wrapper.setDynamicFusionThreshold(float(data_cf["map"].get("fused_dynamic_threshold", 0.5)))
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
        with open(mask_dir / "mask_index_map.txt", "w", encoding="utf-8") as f:
            f.write("mask_id,frame_idx,frame_number,affine_source_frame_idx\n")
        console.print(f"[green][3DGS-MAP][/] Saving masks to: {mask_dir}")

    geo_wrapper.setCamera(
        rgbd_camera.fx_, rgbd_camera.fy_,
        rgbd_camera.cx_, rgbd_camera.cy_,
        rgbd_camera.rows_, rgbd_camera.cols_,
        rgbd_camera.min_depth_, rgbd_camera.max_depth_,
        rgbd_camera.model_,
    )

    return geo_wrapper, gs_visualize, end_frame


def _get_mapping_keyframe_delay(data_cf):
    return max(0, int(data_cf["map"].get("mapping_keyframe_delay", 5)))


def _get_warmup_keyframe_count(data_cf):
    return int(data_cf.get("droidw", {}).get("tracking", {}).get("warmup", 0))


def _delayed_keyframe_mapping_enabled(data_cf, use_gt_pose=False):
    return (
        not use_gt_pose
        and data_cf["map"].get("map_keyframes_only", False)
        and _get_mapping_keyframe_delay(data_cf) > 0
    )


def _map_frame(
    geo_wrapper,
    translation,
    quat,
    depth_img,
    rgb_img,
    gs_visualize,
    dynamic_mask=None,
    dynamic_probability=None,
    timer_summary=None,
):
    """Map a single frame through the 3DGS pipeline.
    
    Args:
        dynamic_mask: optional np.ndarray [H, W] uint8. 255 = dynamic, 0 = static.
        dynamic_probability: optional np.ndarray [H, W] float32. 1 = dynamic, 0 = static.
    """
    timer = timer_summary.measure("3dgs_mapping", sync_cuda=True) if timer_summary is not None else nullcontext()
    with timer:
        geo_wrapper.setCurrPose(translation, quat)
        geo_wrapper.setDepthImage(depth_img)
        geo_wrapper.setRGBImage(rgb_img)
        if dynamic_probability is not None:
            geo_wrapper.setExternalDynamicProbability(dynamic_probability)
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


def _get_dynamic_evidence(
    uncertainty_provider,
    use_droidw_uncertainty,
    dynamic_method,
    frame_idx,
    img_h,
    img_w,
    uncertainty_threshold,
    uncertainty_dilation,
    affine_source_frame_idx=None,
):
    mask = None
    dynamic_probability = None
    if uncertainty_provider is None or not use_droidw_uncertainty:
        return mask, dynamic_probability

    if dynamic_method == "fused_droidw_tsdf":
        dynamic_probability = uncertainty_provider.get_uncertainty_probability(
            frame_idx,
            img_h,
            img_w,
            uncertainty_dilation,
            affine_source_frame_idx=affine_source_frame_idx,
        )
    else:
        mask = uncertainty_provider.get_uncertainty_mask(
            frame_idx,
            img_h,
            img_w,
            uncertainty_threshold,
            uncertainty_dilation,
            affine_source_frame_idx=affine_source_frame_idx,
        )
    return mask, dynamic_probability


def _map_frame_with_outputs(
    geo_wrapper,
    translation,
    quat,
    depth_img,
    rgb_img,
    frame_idx,
    frame_number,
    mapped_frame_id,
    gs_visualize,
    dynamic_method,
    use_droidw_uncertainty,
    uncertainty_provider,
    img_h,
    img_w,
    uncertainty_threshold,
    uncertainty_dilation,
    results_dir,
    rgb_output_dir,
    mask_index_file,
    plot_output_dir,
    comparison_kf_count,
    affine_source_frame_idx=None,
    save_plot=True,
    metric_recorder=None,
    timer_summary=None,
):
    if timer_summary is not None:
        evidence_timer = timer_summary.measure("dynamic_evidence", sync_cuda=True)
    else:
        evidence_timer = nullcontext()
    with evidence_timer:
        mask, dynamic_probability = _get_dynamic_evidence(
            uncertainty_provider,
            use_droidw_uncertainty,
            dynamic_method,
            frame_idx,
            img_h,
            img_w,
            uncertainty_threshold,
            uncertainty_dilation,
            affine_source_frame_idx=affine_source_frame_idx,
        )
    _map_frame(
        geo_wrapper,
        translation,
        quat,
        depth_img,
        rgb_img,
        gs_visualize,
        mask,
        dynamic_probability,
        timer_summary=timer_summary,
    )
    if metric_recorder is not None:
        metric_timer = timer_summary.measure("3dgs_metric_frame_cache", sync_cuda=False) if timer_summary is not None else nullcontext()
        with metric_timer:
            metric_recorder.add_training_frame(
                translation,
                quat,
                frame_idx,
                frame_number,
                mapped_frame_id,
                rgb_img,
            )
    tsdf_residual_img = _get_latest_tsdf_residual_image(geo_wrapper)
    _save_keyframe_rgb(rgb_output_dir, mapped_frame_id, rgb_img)
    _append_mask_index(
        mask_index_file,
        mapped_frame_id,
        frame_idx,
        frame_number,
        affine_source_frame_idx,
    )
    if plot_output_dir is not None and save_plot:
        uncertainty_debug = (
            uncertainty_provider.get_uncertainty_debug_maps(
                frame_idx,
                img_h,
                img_w,
                affine_source_frame_idx=affine_source_frame_idx,
            )
            if uncertainty_provider is not None and use_droidw_uncertainty
            else None
        )
        _save_keyframe_comparison_plot(
            plot_output_dir,
            comparison_kf_count,
            frame_idx,
            mapped_frame_id,
            rgb_img,
            depth_img,
            uncertainty_debug,
            results_dir / "mrhash_mask",
            dynamic_method,
            tsdf_residual_img,
            affine_source_frame_idx=affine_source_frame_idx,
        )
        comparison_kf_count += 1

    return comparison_kf_count


def _append_mask_index(mask_index_file, mask_id, frame_idx, frame_number, affine_source_frame_idx=None):
    if mask_index_file is None:
        return
    affine_source = "" if affine_source_frame_idx is None else int(affine_source_frame_idx)
    with open(mask_index_file, "a", encoding="utf-8") as f:
        f.write(f"{mask_id},{frame_idx},{frame_number},{affine_source}\n")


def _save_keyframe_rgb(output_dir, mask_id, rgb_img):
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_uint8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
    if rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3:
        rgb_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"rgb_{mask_id:06d}.png"), rgb_uint8)


def _get_latest_tsdf_residual_image(geo_wrapper):
    if not hasattr(geo_wrapper, "hasTSDFResidualImage"):
        return None
    if not geo_wrapper.hasTSDFResidualImage():
        return None
    residual_img = np.array(geo_wrapper.getTSDFResidualImage())
    if residual_img.size == 0:
        return None
    return residual_img


def _get_keyframe_plot_dir(data_cf, results_dir):
    return results_dir / data_cf["map"].get("comparison_plots_dir", "plots_final")


def _get_pose_keyframe_set(pose_data):
    if pose_data is None or "keyframe_indices" not in pose_data:
        return None
    return set(int(idx) for idx in pose_data["keyframe_indices"])


def _should_save_keyframe_plot(frame_idx, keyframe_set):
    return keyframe_set is None or int(frame_idx) in keyframe_set


def _float_to_uint8(array, vmin=None, vmax=None, valid_mask=None):
    arr = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(arr)
    if valid_mask is not None:
        valid &= valid_mask

    out = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(valid):
        return out

    vals = arr[valid]
    lo = np.percentile(vals, 2) if vmin is None else float(vmin)
    hi = np.percentile(vals, 98) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6

    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~valid] = 0.0
    out[:] = (norm * 255.0).astype(np.uint8)
    return out


def _float_to_colormap(array, vmin=None, vmax=None, colormap=None, valid_mask=None):
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    gray = _float_to_uint8(array, vmin=vmin, vmax=vmax, valid_mask=valid_mask)
    return cv2.applyColorMap(gray, colormap)


def _float_to_gray_bgr(array, vmin=None, vmax=None, valid_mask=None):
    gray = _float_to_uint8(array, vmin=vmin, vmax=vmax, valid_mask=valid_mask)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _rgb_to_bgr(rgb_img):
    rgb_uint8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
    if rgb_uint8.ndim == 3 and rgb_uint8.shape[2] == 3:
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_GRAY2BGR)


def _depth_to_bgr(depth_img):
    depth = np.asarray(depth_img, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    colormap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)
    depth_bgr = _float_to_colormap(depth, colormap=colormap, valid_mask=valid)
    depth_bgr[~valid] = (0, 0, 0)
    return depth_bgr


def _ensure_bgr(image):
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _read_first_image(paths):
    for path in paths:
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return _ensure_bgr(image)
    return None


def _placeholder_panel(shape, message="N/A"):
    rows, cols = shape[:2]
    panel = np.full((rows, cols, 3), 236, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (cols - 1, rows - 1), (190, 190, 190), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min(cols, rows) / 420.0)
    thickness = max(1, int(round(scale * 2)))
    text_size, _ = cv2.getTextSize(message, font, scale, thickness)
    x = max(0, (cols - text_size[0]) // 2)
    y = max(text_size[1] + 4, (rows + text_size[1]) // 2)
    cv2.putText(panel, message, (x, y), font, scale, (80, 80, 80), thickness, cv2.LINE_AA)
    return panel


def _draw_centered_text(canvas, text, x0, x1, baseline_y, max_scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max_scale
    width = x1 - x0
    while scale > 0.35:
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        if text_size[0] <= width - 8:
            break
        scale -= 0.05
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = x0 + max(0, (width - text_size[0]) // 2)
    cv2.putText(canvas, text, (x, baseline_y), font, scale, color, thickness, cv2.LINE_AA)


def _resize_panel(panel, tile_w, tile_h, interpolation):
    return cv2.resize(_ensure_bgr(panel), (tile_w, tile_h), interpolation=interpolation)


def _compose_keyframe_plot(panels, kf_order, frame_idx, tile_shape, affine_source_frame_idx=None):
    rows, cols = 3, 3
    tile_h, tile_w = tile_shape
    margin = 24
    gap = 20
    title_h = 78
    label_h = 42

    canvas_w = cols * tile_w + (cols - 1) * gap + 2 * margin
    canvas_h = title_h + rows * (label_h + tile_h) + (rows - 1) * gap + margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    title = f"Keyframe idx {kf_order:03d}, Frame idx {frame_idx:05d}"
    if affine_source_frame_idx is not None:
        title += f", Affine src {int(affine_source_frame_idx):05d}"
    _draw_centered_text(canvas, title, 0, canvas_w, 48, 1.25, (35, 35, 35), 2)

    for panel_idx, (label, image, interpolation) in enumerate(panels):
        row = panel_idx // cols
        col = panel_idx % cols
        x0 = margin + col * (tile_w + gap)
        label_y0 = title_h + row * (label_h + tile_h + gap)
        image_y0 = label_y0 + label_h
        _draw_centered_text(
            canvas,
            label,
            x0,
            x0 + tile_w,
            label_y0 + 29,
            0.72,
            (35, 35, 35),
            1,
        )
        tile = _resize_panel(image, tile_w, tile_h, interpolation)
        canvas[image_y0:image_y0 + tile_h, x0:x0 + tile_w] = tile

    return canvas


def _save_keyframe_comparison_plot(
    output_dir,
    kf_order,
    frame_idx,
    mask_id,
    rgb_img,
    depth_img,
    uncertainty_debug,
    mask_dir,
    dynamic_method,
    tsdf_residual_img=None,
    affine_source_frame_idx=None,
):
    base_shape = rgb_img.shape[:2]
    base_h, base_w = base_shape
    tile_w = min(640, base_w)
    tile_h = max(1, int(round(tile_w * base_h / max(base_w, 1))))
    placeholder = _placeholder_panel(base_shape)

    if uncertainty_debug is not None:
        uncertainty = uncertainty_debug["uncertainty"]
        uncertainty_gray = uncertainty_debug["uncertainty_gray"]
        uncertainty_rescaled = uncertainty_debug["uncertainty_rescaled"]
        high_res_scaled = uncertainty_debug["uncertainty_high_rescaled"]
        uncertainty_panel = _float_to_colormap(uncertainty, vmin=0.0, vmax=1.0)
        uncertainty_gray_panel = _float_to_gray_bgr(uncertainty_gray, vmin=0.0, vmax=1.0)
        rescaled_panel = _float_to_colormap(uncertainty_rescaled, vmin=0.0, vmax=10.0)
        high_res_panel = _float_to_colormap(high_res_scaled, vmin=0.0, vmax=10.0)
    else:
        uncertainty_panel = placeholder
        uncertainty_gray_panel = placeholder
        rescaled_panel = placeholder
        high_res_panel = placeholder

    residual_panel = _ensure_bgr(tsdf_residual_img) if tsdf_residual_img is not None else None
    tsdf_panel = _read_first_image([
        mask_dir / f"mask_tsdf_{mask_id:06d}.png",
        mask_dir / "raw" / f"mask_tsdf_{mask_id:06d}.png",
        mask_dir / "raw" / f"mask_{mask_id:06d}.png",
    ])
    fused_panel = _read_first_image([
        mask_dir / f"mask_fused_{mask_id:06d}.png",
        mask_dir / f"mask_{mask_id:06d}.png",
    ])

    residual_panel = residual_panel if residual_panel is not None else _placeholder_panel(base_shape, "No TSDF yet")
    tsdf_panel = tsdf_panel if tsdf_panel is not None else placeholder
    fused_panel = fused_panel if fused_panel is not None else placeholder

    final_mask_label = "Fused Mask" if dynamic_method == "fused_droidw_tsdf" else "Final Mask"
    panels = [
        ("RGB", _rgb_to_bgr(rgb_img), cv2.INTER_AREA),
        ("Depth", _depth_to_bgr(depth_img), cv2.INTER_AREA),
        ("DROID-W Uncertainty", uncertainty_panel, cv2.INTER_NEAREST),
        ("Uncertainty Gray", uncertainty_gray_panel, cv2.INTER_NEAREST),
        ("Rescaled Uncertainty", rescaled_panel, cv2.INTER_NEAREST),
        ("High-Resolution Scaled Uncertainty", high_res_panel, cv2.INTER_AREA),
        ("TSDF Raw Residual", residual_panel, cv2.INTER_AREA),
        ("TSDF Mask", tsdf_panel, cv2.INTER_NEAREST),
        (final_mask_label, fused_panel, cv2.INTER_NEAREST),
    ]

    plot = _compose_keyframe_plot(
        panels,
        kf_order,
        frame_idx,
        (tile_h, tile_w),
        affine_source_frame_idx=affine_source_frame_idx,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_suffix = (
        f"_src_{int(affine_source_frame_idx):05d}"
        if affine_source_frame_idx is not None
        else ""
    )
    output_path = output_dir / f"video_kf_{kf_order:03d}_ts_{frame_idx:05d}{source_suffix}.png"
    cv2.imwrite(str(output_path), plot)


def _save_results(geo_wrapper, results_dir, timestamp, gs_visualize, metric_recorder=None, timer_summary=None):
    """Save mesh, Gaussian splats, and hash/voxel data."""
    if gs_visualize and geo_wrapper.hasGSRenderedImage():
        console.print(
            "[green][3DGS-MAP][/] Mapping complete. Press any key on the image window to close."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print_section("Saving Results", "yellow")

    save_timer = timer_summary.measure("save_3dgs_results", sync_cuda=True) if timer_summary is not None else nullcontext()
    with save_timer:
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

    if metric_recorder is not None:
        metric_timer = timer_summary.measure("3dgs_final_metric_render", sync_cuda=True) if timer_summary is not None else nullcontext()
        with metric_timer:
            metric_recorder.render_final_metrics(geo_wrapper)
        metrics_path = metric_recorder.write()
        console.print(f"[yellow][OUTPUT][/] 3DGS frame metrics: {metrics_path}")

    geo_wrapper.clearBuffers()


def run_online_pipeline(data_cf, reader, timestamp, timer_summary=None):
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
    evaluate_3dgs_metrics = data_cf["map"].get("evaluate_3dgs_metrics", True)
    metric_recorder = GSMetricRecorder(results_dir) if evaluate_3dgs_metrics else None

    # --- Print config info ---
    dynamic_detection = data_cf["map"].get("dynamic_detection", False)
    dynamic_method = data_cf["map"].get("dynamic_method", "tsdf_residual")
    uncertainty_threshold = data_cf["map"].get("droidw_uncertainty_threshold", 0.9)
    uncertainty_dilation = data_cf["map"].get("droidw_uncertainty_dilation", 10)
    use_droidw_uncertainty = dynamic_detection and dynamic_method in ("droidw_uncertainty", "fused_droidw_tsdf")
    map_keyframes_only = data_cf["map"].get("map_keyframes_only", False)
    save_keyframe_plots = data_cf["map"].get("save_keyframe_comparison_plots", False)
    save_dynamic_mask = data_cf["map"].get("save_dynamic_mask", False) or save_keyframe_plots
    mapping_keyframe_delay = _get_mapping_keyframe_delay(data_cf)
    delayed_mapping = _delayed_keyframe_mapping_enabled(data_cf, use_gt_pose=False)
    delayed_start_trigger_kf = max(
        0, droidw_cfg["tracking"]["warmup"] + mapping_keyframe_delay - 1
    )

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
    table.add_row("Keyframe Mapping Delay", str(mapping_keyframe_delay if delayed_mapping else 0))
    table.add_row("GS Visualize", str(gs_visualize))
    table.add_row("Evaluate 3DGS Metrics", str(evaluate_3dgs_metrics))
    table.add_row("Keyframe Comparison Plots", str(save_keyframe_plots))
    console.print(table)

    img_h = data_cf["sensor"]["resolution"][1]
    img_w = data_cf["sensor"]["resolution"][0]

    # --- Main online loop ---
    num_frames = len(reader)
    mapped_frames = 0
    warmup_frame_count = 0
    last_delayed_mapped_kf_buf_idx = -1
    mask_index_file = results_dir / "mrhash_mask" / "mask_index_map.txt" if save_dynamic_mask else None
    rgb_output_dir = results_dir / "rgb"
    rgb_output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[magenta][ONLINE][/] Saving mapped RGB images to: {rgb_output_dir}")
    plot_output_dir = _get_keyframe_plot_dir(data_cf, results_dir) if save_keyframe_plots else None
    if plot_output_dir is not None:
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[magenta][ONLINE][/] Saving keyframe comparison plots to: {plot_output_dir}")
    comparison_kf_count = 0

    console.print(
        f"[magenta][ONLINE][/] Starting online pipeline on {num_frames} frames..."
    )

    for idx in tqdm(range(num_frames), desc="[ONLINE] Track+Map"):
        frame, _, _, depth_img, rgb_img = reader[idx]
        if frame > end_frame:
            break

        # --- Step 1: DROID-W tracking ---
        track_timer = timer_summary.measure("droidw_process_frame", sync_cuda=True) if timer_summary is not None else nullcontext()
        with track_timer:
            result = tracker.process_frame(idx)

        # Free intermediate tracking GPU memory before mapping
        torch.cuda.empty_cache()

        # --- Step 2: 3DGS mapping ---
        if result["just_initialized"] and delayed_mapping:
            console.print(
                f"\n[magenta][ONLINE][/] Warmup done at frame {idx}. "
                f"Delayed mapping will start when keyframe #{delayed_start_trigger_kf + 1} is available."
            )

        elif result["just_initialized"]:
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

            # Sort keyframes by frame index for monotonic interpolation
            _sort = np.argsort(kf_indices)
            kf_indices = kf_indices[_sort].astype(np.float64)
            kf_translations = kf_translations[_sort]
            kf_quaternions = kf_quaternions[_sort]

            # Build SLERP for rotations (quaternions stored as (qx,qy,qz,qw) → scipy as_quat order)
            if len(kf_indices) >= 2:
                _slerp = Slerp(kf_indices, Rotation.from_quat(kf_quaternions))
            else:
                _slerp = None

            def _interp_pose(qry_idx):
                qf = float(qry_idx)
                if _slerp is None or qf <= kf_indices[0]:
                    return kf_translations[0], kf_quaternions[0]
                if qf >= kf_indices[-1]:
                    return kf_translations[-1], kf_quaternions[-1]
                # LERP translation between bracketing keyframes
                hi = int(np.searchsorted(kf_indices, qf, side="right"))
                lo = hi - 1
                w = (qf - kf_indices[lo]) / (kf_indices[hi] - kf_indices[lo])
                t_interp = (1.0 - w) * kf_translations[lo] + w * kf_translations[hi]
                q_interp = _slerp([qf]).as_quat()[0].astype(np.float32)
                return t_interp.astype(np.float32), q_interp

            # Determine which frames to map during warmup
            if map_keyframes_only:
                warmup_map_indices = sorted(int(kf_indices[i]) for i in range(len(kf_indices)))
            else:
                warmup_map_indices = list(range(idx + 1))
            warmup_keyframe_set = set(int(kf_idx) for kf_idx in kf_indices)

            for warmup_idx in warmup_map_indices:
                w_frame, _, _, w_depth, w_rgb = reader[warmup_idx]
                if w_frame > end_frame:
                    break
                # Interpolate pose (SLERP for rotation, LERP for translation)
                t, q = _interp_pose(warmup_idx)
                # Get DROID-W dynamic evidence if using DROID-W-based methods
                mask = None
                dynamic_probability = None
                evidence_timer = timer_summary.measure("dynamic_evidence", sync_cuda=True) if timer_summary is not None else nullcontext()
                with evidence_timer:
                    if use_droidw_uncertainty:
                        if dynamic_method == "fused_droidw_tsdf":
                            dynamic_probability = tracker.get_uncertainty_probability(warmup_idx, img_h, img_w, uncertainty_dilation)
                        else:
                            mask = tracker.get_uncertainty_mask(warmup_idx, img_h, img_w, uncertainty_threshold, uncertainty_dilation)
                _map_frame(geo_wrapper, t, q, w_depth, w_rgb, gs_visualize, mask, dynamic_probability, timer_summary=timer_summary)
                if metric_recorder is not None:
                    metric_timer = timer_summary.measure("3dgs_metric_frame_cache", sync_cuda=False) if timer_summary is not None else nullcontext()
                    with metric_timer:
                        metric_recorder.add_training_frame(
                            t,
                            q,
                            warmup_idx,
                            w_frame,
                            mapped_frames,
                            w_rgb,
                        )
                tsdf_residual_img = _get_latest_tsdf_residual_image(geo_wrapper)
                _save_keyframe_rgb(rgb_output_dir, mapped_frames, w_rgb)
                _append_mask_index(mask_index_file, mapped_frames, warmup_idx, w_frame)
                if plot_output_dir is not None and warmup_idx in warmup_keyframe_set:
                    uncertainty_debug = (
                        tracker.get_uncertainty_debug_maps(warmup_idx, img_h, img_w)
                        if use_droidw_uncertainty
                        else None
                    )
                    _save_keyframe_comparison_plot(
                        plot_output_dir,
                        comparison_kf_count,
                        warmup_idx,
                        mapped_frames,
                        w_rgb,
                        w_depth,
                        uncertainty_debug,
                        results_dir / "mrhash_mask",
                        dynamic_method,
                        tsdf_residual_img,
                    )
                    comparison_kf_count += 1
                mapped_frames += 1

        elif result["pose_available"]:
            if delayed_mapping:
                if not result["is_keyframe"]:
                    continue

                current_kf_buf_idx = result["current_keyframe_buffer_idx"]
                if current_kf_buf_idx < delayed_start_trigger_kf:
                    continue

                affine_source_frame_idx = tracker.get_keyframe_frame_idx(current_kf_buf_idx)
                if affine_source_frame_idx is None:
                    continue

                eligible_target_kf_buf_idx = current_kf_buf_idx - mapping_keyframe_delay
                while last_delayed_mapped_kf_buf_idx < eligible_target_kf_buf_idx:
                    target_kf_buf_idx = last_delayed_mapped_kf_buf_idx + 1
                    target_frame_idx = tracker.get_keyframe_frame_idx(target_kf_buf_idx)
                    if target_frame_idx is None:
                        last_delayed_mapped_kf_buf_idx = target_kf_buf_idx
                        continue

                    t, q = tracker.get_keyframe_pose_by_buffer_index(target_kf_buf_idx)
                    if t is None:
                        last_delayed_mapped_kf_buf_idx = target_kf_buf_idx
                        continue

                    target_frame, _, _, target_depth, target_rgb = reader[target_frame_idx]
                    if target_frame > end_frame:
                        break

                    comparison_kf_count = _map_frame_with_outputs(
                        geo_wrapper,
                        t,
                        q,
                        target_depth,
                        target_rgb,
                        target_frame_idx,
                        target_frame,
                        mapped_frames,
                        gs_visualize,
                        dynamic_method,
                        use_droidw_uncertainty,
                        tracker,
                        img_h,
                        img_w,
                        uncertainty_threshold,
                        uncertainty_dilation,
                        results_dir,
                        rgb_output_dir,
                        mask_index_file,
                        plot_output_dir,
                        comparison_kf_count,
                        affine_source_frame_idx=affine_source_frame_idx,
                        save_plot=True,
                        metric_recorder=metric_recorder,
                        timer_summary=timer_summary,
                    )
                    mapped_frames += 1
                    last_delayed_mapped_kf_buf_idx = target_kf_buf_idx
                continue

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
            comparison_kf_count = _map_frame_with_outputs(
                geo_wrapper,
                result["translation"],
                result["quaternion"],
                depth_img,
                rgb_img,
                idx,
                frame,
                mapped_frames,
                gs_visualize,
                dynamic_method,
                use_droidw_uncertainty,
                tracker,
                img_h,
                img_w,
                uncertainty_threshold,
                uncertainty_dilation,
                results_dir,
                rgb_output_dir,
                mask_index_file,
                plot_output_dir,
                comparison_kf_count,
                save_plot=result["is_keyframe"],
                metric_recorder=metric_recorder,
                timer_summary=timer_summary,
            )
            mapped_frames += 1
        else:
            warmup_frame_count += 1

    if delayed_mapping:
        final_kf_buf_idx = tracker.video.counter.value - 1
        if final_kf_buf_idx >= 0 and last_delayed_mapped_kf_buf_idx < final_kf_buf_idx:
            affine_source_frame_idx = tracker.get_keyframe_frame_idx(final_kf_buf_idx)
            console.print(
                "\n[magenta][ONLINE][/] Flushing delayed tail keyframes without "
                f"delay constraint: #{last_delayed_mapped_kf_buf_idx + 2}-#{final_kf_buf_idx + 1}"
            )
            while last_delayed_mapped_kf_buf_idx < final_kf_buf_idx:
                target_kf_buf_idx = last_delayed_mapped_kf_buf_idx + 1
                target_frame_idx = tracker.get_keyframe_frame_idx(target_kf_buf_idx)
                if target_frame_idx is None or affine_source_frame_idx is None:
                    last_delayed_mapped_kf_buf_idx = target_kf_buf_idx
                    continue

                t, q = tracker.get_keyframe_pose_by_buffer_index(target_kf_buf_idx)
                if t is None:
                    last_delayed_mapped_kf_buf_idx = target_kf_buf_idx
                    continue

                target_frame, _, _, target_depth, target_rgb = reader[target_frame_idx]
                if target_frame > end_frame:
                    break

                comparison_kf_count = _map_frame_with_outputs(
                    geo_wrapper,
                    t,
                    q,
                    target_depth,
                    target_rgb,
                    target_frame_idx,
                    target_frame,
                    mapped_frames,
                    gs_visualize,
                    dynamic_method,
                    use_droidw_uncertainty,
                    tracker,
                    img_h,
                    img_w,
                    uncertainty_threshold,
                    uncertainty_dilation,
                    results_dir,
                    rgb_output_dir,
                    mask_index_file,
                    plot_output_dir,
                    comparison_kf_count,
                    affine_source_frame_idx=affine_source_frame_idx,
                    save_plot=True,
                    metric_recorder=metric_recorder,
                    timer_summary=timer_summary,
                )
                mapped_frames += 1
                last_delayed_mapped_kf_buf_idx = target_kf_buf_idx

    console.print(
        f"\n[magenta][ONLINE][/] Pipeline complete: "
        f"{mapped_frames} frames mapped, "
        f"{warmup_frame_count} frames in warmup phase"
    )

    # --- Save DROID-W estimated poses ---
    output_dir = Path(data_cf["results_path"]) / "droidw"
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_timer = timer_summary.measure("save_droidw_poses", sync_cuda=False) if timer_summary is not None else nullcontext()
    with pose_timer:
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

    eval_timer = timer_summary.measure("localization_accuracy_eval", sync_cuda=False) if timer_summary is not None else nullcontext()
    with eval_timer:
        write_localization_accuracy(data_cf, reader, pose_data)

    # --- Finalize tracker ---
    finalize_timer = timer_summary.measure("droidw_finalize", sync_cuda=True) if timer_summary is not None else nullcontext()
    with finalize_timer:
        tracker.finalize()

    # --- Save 3DGS results ---
    _save_results(geo_wrapper, results_dir, timestamp, gs_visualize, metric_recorder, timer_summary)


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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(data_cf["results_path"]) / timestamp
    data_cf["results_path"] = str(results_dir)
    use_gt_pose = data_cf.get("use_gt_pose", False)

    if not data_path.exists():
        console.print(f"[red]Error: Data path {data_path} does not exist!")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)
    timer_summary = TimerSummary()

    # Save a copy of the config
    copied_config = results_dir / f"{timestamp}_{config.name}"
    shutil.copy(config, copied_config)

    # Print config summary
    print_config_summary(data_cf, use_gt_pose)

    # --- Load Data ---
    print_section("Loading TUM RGB-D Dataset", "yellow")

    sensor = data_cf["sensor"]
    with timer_summary.measure("dataset_load", sync_cuda=False):
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
        run_online_pipeline(data_cf, reader, timestamp, timer_summary)
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
            pose_data, slam_wrapper = run_droidw_tracking(data_cf, reader, timer_summary)
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
        run_dynamic_3dgs_mapping(data_cf, reader, pose_data, use_gt_pose, timestamp, slam_wrapper, timer_summary)

    timer_summary_path = results_dir / "timer_summary.txt"
    timer_summary.write(timer_summary_path)
    console.print(f"[yellow][OUTPUT][/] Timer summary: {timer_summary_path}")

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
