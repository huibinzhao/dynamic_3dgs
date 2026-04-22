"""
DROID-W SLAM wrapper for integration with dynamic_3dgs.

This module provides a simplified SLAM interface that only runs
DROID-W's visual odometry tracking and extracts estimated poses
for use by the dynamic_3dgs mapping pipeline.

Supports both offline (batch) and online (per-frame) modes.
"""

import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import OrderedDict
from scipy.spatial.transform import Rotation

# Import from DROID-W (added to sys.path by __init__.py)
from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_fill
from src.utils.datasets import BaseDataset, RGB_NoPose
from src.motion_filter import MotionFilter
from src.frontend import Frontend
from src.tracker import Tracker
from src.backend import Backend
from src.utils.sys_timer import timer
from torch.utils.tensorboard import SummaryWriter
from lietorch import SE3


class IntegratedPrinter:
    """
    A unified printer that displays both DROID-W and dynamic_3dgs messages
    with clear section labels and colored output.
    """

    def __init__(self, total_img_num):
        self.msg_lock = mp.Lock()
        self.msg_queue = mp.Queue()
        self.progress_counter = mp.Value("i", 0)
        self.total_img_num = total_img_num
        # NOTE: Do NOT store the Process as self._process — it makes
        # IntegratedPrinter unpicklable, which breaks spawn-based
        # multiprocessing when this object is referenced by a child process.
        _printer_proc = mp.Process(
            target=self._printer_process, args=(total_img_num,)
        )
        _printer_proc.start()

    def print(self, msg: str, color=None):
        from colorama import Style

        if color is not None:
            msg_prefix = self._get_prefix(color)
            msg = msg_prefix + str(msg) + Style.RESET_ALL
        with self.msg_lock:
            self.msg_queue.put(msg)

    def print_dynamic3dgs(self, msg: str):
        """Print a message tagged as dynamic_3dgs."""
        from colorama import Fore, Style

        tagged = Fore.GREEN + "[3DGS-MAP] " + Style.RESET_ALL + str(msg)
        with self.msg_lock:
            self.msg_queue.put(tagged)

    def print_droidw(self, msg: str):
        """Print a message tagged as DROID-W."""
        from colorama import Fore, Style

        tagged = Fore.BLUE + "[DROID-W] " + Style.RESET_ALL + str(msg)
        with self.msg_lock:
            self.msg_queue.put(tagged)

    def print_system(self, msg: str):
        """Print a system-level message."""
        from colorama import Fore, Style

        tagged = Fore.YELLOW + "[SYSTEM] " + Style.RESET_ALL + str(msg)
        with self.msg_lock:
            self.msg_queue.put(tagged)

    def update_pbar(self):
        with self.msg_lock:
            self.progress_counter.value += 1
            self.msg_queue.put("PROGRESS")

    def pbar_ready(self):
        with self.msg_lock:
            self.msg_queue.put("READY")

    def _get_prefix(self, color):
        from colorama import Fore, Style

        prefix_map = {
            Fore.CYAN: Fore.CYAN + "[MAPPER] " + Style.RESET_ALL,
            Fore.BLUE: Fore.BLUE + "[DROID-W TRACKER] " + Style.RESET_ALL,
            Fore.YELLOW: Fore.YELLOW + "[INFO] " + Style.RESET_ALL,
            Fore.RED: Fore.RED + "[ERROR] " + Style.RESET_ALL,
            Fore.GREEN: Fore.GREEN + "[3DGS] " + Style.RESET_ALL,
            Fore.MAGENTA: Fore.MAGENTA + "[EVAL] " + Style.RESET_ALL,
            "yellow": Fore.YELLOW + "[MESH] " + Style.RESET_ALL,
        }
        return prefix_map.get(color, Style.RESET_ALL)

    def _printer_process(self, total_img_num):
        from tqdm import tqdm
        from colorama import Fore, Style

        # Print messages before progress bar is ready
        while True:
            message = self.msg_queue.get()
            if message == "READY":
                break
            elif message == "DONE":
                return
            else:
                print(message)

        # Progress bar phase
        with tqdm(total=total_img_num, desc="DROID-W Tracking") as pbar:
            while self.progress_counter.value < total_img_num:
                message = self.msg_queue.get()
                if message == "DONE":
                    break
                elif message.startswith("PROGRESS"):
                    with self.msg_lock:
                        completed = self.progress_counter.value
                    pbar.set_description(
                        Fore.BLUE + "[DROID-W Tracking]" + Style.RESET_ALL
                    )
                    pbar.n = completed
                    pbar.refresh()
                else:
                    pbar.write(message)

        # Post-tracking messages
        while True:
            message = self.msg_queue.get()
            if message == "DONE":
                break
            else:
                print(message)

    def terminate(self):
        self.msg_queue.put("DONE")


class DroidWSLAMWrapper:
    """
    Wrapper around DROID-W SLAM for integration with dynamic_3dgs.

    Runs DROID-W tracking only (no GS mapping), then extracts
    estimated camera poses for each frame. Final Global BA and
    Full Trajectory Filling are placed here but commented out
    for future activation.
    """

    def __init__(self, droidw_cfg, stream: BaseDataset, printer: IntegratedPrinter):
        self.cfg = droidw_cfg
        self.device = droidw_cfg["device"]
        self.verbose = droidw_cfg["verbose"]
        self.save_dir = droidw_cfg["data"]["output"] + "/" + droidw_cfg["scene"]
        self.printer = printer
        self.stream = stream

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(droidw_cfg)

        # Build the DROID network
        self.droid_net = DroidNet()
        self._load_pretrained(droidw_cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        # Video buffer (shared memory for poses, depths, etc.)
        self.video = DepthVideo(droidw_cfg, self.printer)
        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # Trajectory filler for non-keyframe poses
        self.traj_filler = PoseTrajectoryFiller(
            cfg=droidw_cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker = None

    def _load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        # Resolve relative paths against the droidw directory
        if not os.path.isabs(droid_pretrained):
            _droidw_dir = os.path.dirname(os.path.abspath(__file__))
            droid_pretrained = os.path.join(_droidw_dir, droid_pretrained)
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print_droidw(
            f"Loaded pretrained checkpoint from {droid_pretrained}"
        )

    def _tracking_process(self, pipe):
        """Run DROID-W tracking in a subprocess."""
        for file in os.listdir(self.save_dir):
            if file.startswith("events.out.tfevents."):
                os.remove(os.path.join(self.save_dir, file))

        event_writer = SummaryWriter(self.save_dir)
        self.tracker = Tracker(self, pipe, event_writer)
        self.printer.print_droidw("Tracking started")
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print_droidw("Tracking completed")

        # --- Post-tracking pose extraction ---
        self._extract_and_save_poses()

    def _extract_and_save_poses(self):
        """
        Extract poses after tracking completes.
        Final Global BA and Full Trajectory Filling are commented out
        for now. Uncomment them when ready to use.
        """

        # ============================================================
        # [COMMENTED OUT] Final Global BA
        # Uncomment the following block to enable final global bundle
        # adjustment before extracting poses for dynamic_3dgs.
        # ============================================================
        # self.printer.print_droidw("Running Final Global BA...")
        # metric_depth_reg_activated = self.video.metric_depth_reg
        # if metric_depth_reg_activated:
        #     self.video.metric_depth_reg = False
        # ba = Backend(self.droid_net, self.video, self.cfg)
        # torch.cuda.empty_cache()
        # ba.dense_ba(7, enable_udba=self.cfg['tracking']['frontend']['enable_opt_dyn_mask'])
        # torch.cuda.empty_cache()
        # ba.dense_ba(12, enable_udba=self.cfg['tracking']['frontend']['enable_opt_dyn_mask'])
        # self.printer.print_droidw("Final Global BA completed")
        # if metric_depth_reg_activated:
        #     self.video.metric_depth_reg = True

        # ============================================================
        # [COMMENTED OUT] Full Trajectory Filling
        # Uncomment the following block to fill non-keyframe poses
        # using DROID-W's trajectory filler before feeding poses
        # to dynamic_3dgs.
        # ============================================================
        # self.printer.print_droidw("Running Full Trajectory Filling...")
        # self.traj_filler.setup_feature_extractor()
        # traj_est = full_traj_fill(
        #     self.traj_filler, None, self.stream, fast_mode=True
        # )
        # self.printer.print_droidw("Full Trajectory Filling completed")

        # Save keyframe video data
        self.video.save_video(f"{self.save_dir}/video.npz")
        self.printer.print_droidw(f"Video data saved to {self.save_dir}/video.npz")

    def run_tracking(self):
        """
        Run DROID-W tracking and return estimated poses for all frames.

        Returns:
            all_poses: dict with keys:
                'keyframe_indices': np.ndarray [K] frame indices of keyframes
                'keyframe_poses_c2w': np.ndarray [K, 4, 4] c2w poses for keyframes
                'all_translations': np.ndarray [N, 3] translations for all frames
                'all_quaternions': np.ndarray [N, 4] quaternions (qx,qy,qz,qw) for all frames
                'num_keyframes': int
                'num_frames': int
        """
        self.printer.print_system(
            f"Starting DROID-W tracking on {len(self.stream)} frames..."
        )

        # spawn start method is already set globally in the main script
        m_pipe, t_pipe = mp.Pipe()

        processes = [
            mp.Process(target=self._tracking_process, args=(t_pipe,)),
        ]
        self.num_running_thread[0] = len(processes)
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self.printer.terminate()

        # Extract poses from the video buffer
        return self._get_all_poses()

    def _get_all_poses(self):
        """
        Extract poses from DROID-W's video buffer and interpolate
        for non-keyframe frames.

        Returns pose as (translation, quaternion) for every frame
        in the input stream.
        """
        kf_num = self.video.counter.value
        num_frames = len(self.stream)

        # Get keyframe data
        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()
        # video.poses are stored in w2c SE3 format; invert to get c2w
        kf_poses_c2w = (
            SE3(self.video.poses[:kf_num].clone())
            .inv()
            .matrix()
            .data.cpu()
            .numpy()
        )  # [K, 4, 4]

        self.printer.print_droidw(
            f"Extracted {kf_num} keyframe poses from {num_frames} total frames"
        )

        # Interpolate poses for all frames
        all_poses_4x4 = self._interpolate_poses(
            kf_timestamps, kf_poses_c2w, num_frames
        )

        # Convert to (translation, quaternion) format
        all_translations = np.zeros((num_frames, 3), dtype=np.float32)
        all_quaternions = np.zeros((num_frames, 4), dtype=np.float32)

        for i in range(num_frames):
            T = all_poses_4x4[i]
            all_translations[i] = T[:3, 3].astype(np.float32)
            r = Rotation.from_matrix(T[:3, :3])
            # quat in (qx, qy, qz, qw) format - same as TUM
            all_quaternions[i] = r.as_quat().astype(np.float32)

        return {
            "keyframe_indices": kf_timestamps,
            "keyframe_poses_c2w": kf_poses_c2w,
            "all_translations": all_translations,
            "all_quaternions": all_quaternions,
            "all_poses_4x4": all_poses_4x4,
            "num_keyframes": kf_num,
            "num_frames": num_frames,
        }

    def get_uncertainty_mask(self, frame_idx, img_h, img_w, threshold=0.9, dilation_size=10):
        """
        Extract DROID-W uncertainty-based dynamic mask for a given frame.

        Args:
            frame_idx: frame index (0-based)
            img_h: target image height (full resolution)
            img_w: target image width (full resolution)
            threshold: mask value threshold. Pixels below → dynamic.
            dilation_size: radius of elliptical dilation kernel (0 = no dilation).

        Returns:
            np.ndarray [img_h, img_w] dtype=uint8. 255 = dynamic, 0 = static.
            Returns None if unavailable.
        """
        import torch.nn.functional as F
        import cv2

        kf_num = self.video.counter.value
        if kf_num == 0:
            return None

        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()

        kf_match = np.where(kf_timestamps == frame_idx)[0]
        if len(kf_match) > 0:
            buf_idx = kf_match[0]
        else:
            diffs = kf_timestamps - frame_idx
            past = diffs[diffs <= 0]
            if len(past) > 0:
                buf_idx = np.where(diffs == past.max())[0][0]
            else:
                buf_idx = np.argmin(np.abs(diffs))

        uncer = self.video.uncertainties[buf_idx]

        # Bilinear interpolate raw uncertainty to full resolution first
        uncer_upscaled = F.interpolate(
            uncer.unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze()

        # Then apply linear scaling and inversion
        uncer_rescaled = torch.clamp(45.0 * uncer_upscaled - 35.0, min=0.1)
        mask_fullres = torch.clamp(1.0 / uncer_rescaled, min=0.0, max=1.0)

        dynamic_mask = (mask_fullres < threshold).cpu().numpy().astype(np.uint8) * 255

        return dynamic_mask

    def _interpolate_poses(self, kf_indices, kf_poses, num_frames):
        """
        Simple pose interpolation for non-keyframe frames.
        Uses SLERP for rotation and LERP for translation.

        Args:
            kf_indices: [K] sorted keyframe frame indices
            kf_poses: [K, 4, 4] c2w poses for keyframes
            num_frames: total number of frames

        Returns:
            all_poses: [N, 4, 4] interpolated c2w poses
        """
        all_poses = np.zeros((num_frames, 4, 4), dtype=np.float64)

        # Sort keyframes by index
        sort_idx = np.argsort(kf_indices)
        kf_indices = kf_indices[sort_idx]
        kf_poses = kf_poses[sort_idx]

        # For frames before the first keyframe, use the first keyframe pose
        for i in range(0, kf_indices[0]):
            all_poses[i] = kf_poses[0]

        # For frames after the last keyframe, use the last keyframe pose
        for i in range(kf_indices[-1] + 1, num_frames):
            all_poses[i] = kf_poses[-1]

        # Set keyframe poses exactly
        for k in range(len(kf_indices)):
            idx = kf_indices[k]
            if idx < num_frames:
                all_poses[idx] = kf_poses[k]

        # Interpolate between consecutive keyframes
        for k in range(len(kf_indices) - 1):
            idx_start = kf_indices[k]
            idx_end = kf_indices[k + 1]

            if idx_end - idx_start <= 1:
                continue

            t_start = kf_poses[k][:3, 3]
            t_end = kf_poses[k + 1][:3, 3]
            r_start = Rotation.from_matrix(kf_poses[k][:3, :3])
            r_end = Rotation.from_matrix(kf_poses[k + 1][:3, :3])

            from scipy.spatial.transform import Slerp

            key_rots = Rotation.concatenate([r_start, r_end])
            slerp = Slerp([0, 1], key_rots)

            for i in range(idx_start + 1, idx_end):
                alpha = (i - idx_start) / (idx_end - idx_start)
                # SLERP for rotation
                r_interp = slerp(alpha)
                # LERP for translation
                t_interp = (1 - alpha) * t_start + alpha * t_end
                all_poses[i] = np.eye(4)
                all_poses[i][:3, :3] = r_interp.as_matrix()
                all_poses[i][:3, 3] = t_interp

        return all_poses


class SimplePrinter:
    """
    A lightweight printer for online mode that prints directly
    without multiprocessing queues or subprocesses.
    """

    def __init__(self):
        from colorama import Fore, Style
        self._Fore = Fore
        self._Style = Style

    def print(self, msg: str, color=None):
        if color is not None:
            print(f"{color}{msg}{self._Style.RESET_ALL}")
        else:
            print(msg)

    def print_droidw(self, msg: str):
        print(f"{self._Fore.BLUE}[DROID-W]{self._Style.RESET_ALL} {msg}")

    def print_dynamic3dgs(self, msg: str):
        print(f"{self._Fore.GREEN}[3DGS-MAP]{self._Style.RESET_ALL} {msg}")

    def print_system(self, msg: str):
        print(f"{self._Fore.YELLOW}[SYSTEM]{self._Style.RESET_ALL} {msg}")

    def update_pbar(self):
        pass  # No progress bar in online mode

    def pbar_ready(self):
        pass


class OnlineDroidWTracker:
    """
    Online per-frame DROID-W tracker for tight coupling with 3DGS mapping.

    Unlike DroidWSLAMWrapper which runs tracking as a batch subprocess,
    this class processes frames one at a time in the calling process,
    allowing the caller to interleave tracking with 3DGS mapping.

    Usage:
        tracker = OnlineDroidWTracker(droidw_cfg, stream)
        for i in range(len(stream)):
            result = tracker.process_frame(i)
            if result['pose_available']:
                # Use result['translation'], result['quaternion'] for 3DGS
        tracker.finalize()
    """

    def __init__(self, droidw_cfg, stream: BaseDataset):
        self.cfg = droidw_cfg
        self.device = droidw_cfg["device"]
        self.verbose = droidw_cfg["verbose"]
        self.save_dir = droidw_cfg["data"]["output"] + "/" + droidw_cfg["scene"]
        self.stream = stream

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(droidw_cfg)

        # Simple printer (no multiprocessing) — must be created before anything that uses it
        self.printer = SimplePrinter()

        # Build DroidNet
        self.droid_net = DroidNet()
        self._load_pretrained(droidw_cfg)
        self.droid_net.to(self.device).eval()

        # Video buffer
        self.video = DepthVideo(droidw_cfg, self.printer)

        # Tracking components
        filter_thresh = droidw_cfg["tracking"]["motion_filter"]["thresh"]
        self.motion_filter = MotionFilter(
            self.droid_net, self.video, droidw_cfg,
            thresh=filter_thresh, device=self.device,
        )
        self.frontend = Frontend(self.droid_net, self.video, droidw_cfg)

        # Backend for online BA
        self.enable_online_ba = droidw_cfg["tracking"]["frontend"]["enable_online_ba"]
        self.online_ba = Backend(self.droid_net, self.video, droidw_cfg)
        self.ba_freq = droidw_cfg["tracking"]["backend"]["ba_freq"]
        self.finish_first_online_ba = False

        self.intrinsic = stream.get_intrinsic()

        # Tracking state
        self.prev_kf_idx = 0
        self.curr_kf_idx = 0
        self.prev_ba_idx = 0
        self._warmup_complete = False

        # SummaryWriter for tensorboard
        for fname in os.listdir(self.save_dir):
            if fname.startswith("events.out.tfevents."):
                os.remove(os.path.join(self.save_dir, fname))
        self.event_writer = SummaryWriter(self.save_dir)

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        self.printer.print_droidw(
            f"Online tracker initialized (warmup={droidw_cfg['tracking']['warmup']})"
        )

    def _load_pretrained(self, cfg):
        """Load pretrained DROID network weights."""
        droid_pretrained = cfg["tracking"]["pretrained"]
        if not os.path.isabs(droid_pretrained):
            _droidw_dir = os.path.dirname(os.path.abspath(__file__))
            droid_pretrained = os.path.join(_droidw_dir, droid_pretrained)
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print_droidw(f"Loaded pretrained checkpoint from {droid_pretrained}")

    @property
    def is_initialized(self):
        """Whether DROID-W has completed warmup and initialization."""
        return self._warmup_complete

    def process_frame(self, frame_idx):
        """
        Process one frame through DROID-W tracking.

        Args:
            frame_idx: index into the stream dataset (0-based)

        Returns:
            dict with:
                'pose_available': bool - True if a pose can be provided
                'is_keyframe': bool - True if this frame became a keyframe
                'just_initialized': bool - True if warmup just completed
                'translation': np.ndarray [3] c2w translation (only if pose_available)
                'quaternion': np.ndarray [4] (qx,qy,qz,qw) c2w (only if pose_available)
                'warmup_poses': list of (frame_idx, translation, quaternion) for all
                                warmup keyframes (only when just_initialized=True)
        """
        timestamp, image, depth, _ = self.stream[frame_idx]

        with torch.no_grad():
            starting_count = self.video.counter.value

            with timer.section("Tracking"):
                force_to_add_keyframe = self.motion_filter.track(
                    timestamp, image, self.intrinsic, depth=depth
                )
                self.frontend(force_to_add_keyframe, self.event_writer)

            torch.cuda.empty_cache()

        self.curr_kf_idx = self.video.counter.value - 1
        new_keyframe = starting_count < self.video.counter.value
        just_initialized = False

        if new_keyframe and self.frontend.is_initialized:
            if self.video.counter.value == self.frontend.warmup:
                # Warmup just completed - run 2nd stage initialization
                self.frontend.initialize_second_stage(self.event_writer)
                self._warmup_complete = True
                just_initialized = True
                self.printer.print_droidw(
                    f"Warmup complete ({self.video.counter.value} keyframes). "
                    f"Online tracking active."
                )
            else:
                # Normal post-warmup keyframe
                if (
                    self.enable_online_ba
                    and self.curr_kf_idx >= self.prev_ba_idx + self.ba_freq
                ):
                    with timer.section("Online BA"):
                        if not self.finish_first_online_ba:
                            self.online_ba.dense_ba(
                                2,
                                enable_update_uncer=self.frontend.enable_opt_dyn_mask,
                                enable_udba=self.frontend.enable_opt_dyn_mask,
                            )
                            self.finish_first_online_ba = True
                        else:
                            self.online_ba.dense_ba(
                                2,
                                enable_update_uncer=False,
                                enable_udba=self.frontend.enable_opt_dyn_mask,
                            )
                    self.prev_ba_idx = self.curr_kf_idx

            self.prev_kf_idx = self.curr_kf_idx

        # Build result
        result = {
            "pose_available": False,
            "is_keyframe": new_keyframe,
            "just_initialized": just_initialized,
        }

        if just_initialized:
            # Return poses for all warmup keyframes
            warmup_poses = self._get_warmup_keyframe_poses()
            result["warmup_poses"] = warmup_poses
            result["pose_available"] = True
            # Also return current frame pose
            t, q = self._get_latest_pose()
            result["translation"] = t
            result["quaternion"] = q
        elif self._warmup_complete:
            # Post-warmup: always provide a pose
            t, q = self._get_pose_for_frame(frame_idx)
            result["pose_available"] = True
            result["translation"] = t
            result["quaternion"] = q

        return result

    def _get_latest_pose(self):
        """
        Get the pose of the most recent keyframe in c2w format.

        Returns:
            translation: np.ndarray [3]
            quaternion: np.ndarray [4] (qx, qy, qz, qw)
        """
        kf_idx = self.video.counter.value - 1
        c2w = (
            SE3(self.video.poses[kf_idx : kf_idx + 1].clone())
            .inv()
            .matrix()
            .data.cpu()
            .numpy()[0]
        )
        translation = c2w[:3, 3].astype(np.float32)
        quaternion = Rotation.from_matrix(c2w[:3, :3]).as_quat().astype(np.float32)
        return translation, quaternion

    def _get_pose_for_frame(self, frame_idx):
        """
        Get the best available c2w pose for a given frame index.

        For keyframes: returns their optimized pose from the video buffer.
        For non-keyframes: returns the nearest keyframe's pose.

        Returns:
            translation: np.ndarray [3]
            quaternion: np.ndarray [4] (qx, qy, qz, qw)
        """
        kf_num = self.video.counter.value
        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()

        # Check if this frame is a keyframe
        kf_match = np.where(kf_timestamps == frame_idx)[0]
        if len(kf_match) > 0:
            buf_idx = kf_match[0]
        else:
            # Find nearest keyframe (prefer previous)
            diffs = kf_timestamps - frame_idx
            past = diffs[diffs <= 0]
            if len(past) > 0:
                buf_idx = np.where(diffs == past.max())[0][0]
            else:
                buf_idx = np.argmin(np.abs(diffs))

        c2w = (
            SE3(self.video.poses[buf_idx : buf_idx + 1].clone())
            .inv()
            .matrix()
            .data.cpu()
            .numpy()[0]
        )
        translation = c2w[:3, 3].astype(np.float32)
        quaternion = Rotation.from_matrix(c2w[:3, :3]).as_quat().astype(np.float32)
        return translation, quaternion

    def _get_warmup_keyframe_poses(self):
        """
        Get poses for all warmup keyframes.

        Returns:
            list of (frame_idx, translation, quaternion) tuples
        """
        kf_num = self.video.counter.value
        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()
        kf_poses_c2w = (
            SE3(self.video.poses[:kf_num].clone())
            .inv()
            .matrix()
            .data.cpu()
            .numpy()
        )

        poses = []
        for i in range(kf_num):
            t = kf_poses_c2w[i][:3, 3].astype(np.float32)
            q = Rotation.from_matrix(kf_poses_c2w[i][:3, :3]).as_quat().astype(np.float32)
            poses.append((int(kf_timestamps[i]), t, q))
        return poses

    def get_all_poses(self):
        """
        Get poses for all keyframes currently in the buffer.
        Same format as DroidWSLAMWrapper._get_all_poses() for compatibility.
        """
        kf_num = self.video.counter.value
        num_frames = len(self.stream)
        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()
        kf_poses_c2w = (
            SE3(self.video.poses[:kf_num].clone())
            .inv()
            .matrix()
            .data.cpu()
            .numpy()
        )

        all_poses_4x4 = DroidWSLAMWrapper._interpolate_poses(
            None, kf_timestamps, kf_poses_c2w, num_frames
        )

        all_translations = np.zeros((num_frames, 3), dtype=np.float32)
        all_quaternions = np.zeros((num_frames, 4), dtype=np.float32)
        for i in range(num_frames):
            T = all_poses_4x4[i]
            all_translations[i] = T[:3, 3].astype(np.float32)
            all_quaternions[i] = Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32)

        return {
            "keyframe_indices": kf_timestamps,
            "all_translations": all_translations,
            "all_quaternions": all_quaternions,
            "all_poses_4x4": all_poses_4x4,
            "num_keyframes": kf_num,
            "num_frames": num_frames,
        }

    def save_video(self, path=None):
        """Save the video buffer to disk."""
        if path is None:
            path = f"{self.save_dir}/video.npz"
        self.video.save_video(path)
        self.printer.print_droidw(f"Video data saved to {path}")

    def finalize(self):
        """
        Finalize tracking: save video buffer and clean up.
        """
        self.save_video()
        self.event_writer.close()
        self.printer.print_droidw("Online tracker finalized.")

    def get_uncertainty_mask(self, frame_idx, img_h, img_w, threshold=0.9, dilation_size=10):
        """
        Extract DROID-W uncertainty-based dynamic mask for a given frame.

        The uncertainty map from DROID-W (at 1/8 resolution) is upscaled to
        full image resolution and thresholded to produce a binary mask where
        True = dynamic (should be masked out).

        Args:
            frame_idx: frame index (0-based) in the input stream
            img_h: target image height (full resolution)
            img_w: target image width (full resolution)
            threshold: mask threshold in [0, 1]. Pixels with mask value < threshold
                       are considered dynamic. Default 0.9.
            dilation_size: radius of elliptical dilation kernel (0 = no dilation).

        Returns:
            np.ndarray [img_h, img_w] dtype=uint8. 255 = dynamic, 0 = static.
            Returns None if uncertainty is not available for this frame.
        """
        import torch.nn.functional as F
        import cv2

        kf_num = self.video.counter.value
        if kf_num == 0:
            return None

        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()

        # Find the corresponding keyframe buffer index for this frame
        kf_match = np.where(kf_timestamps == frame_idx)[0]
        if len(kf_match) > 0:
            buf_idx = kf_match[0]
        else:
            # Find nearest keyframe
            diffs = kf_timestamps - frame_idx
            past = diffs[diffs <= 0]
            if len(past) > 0:
                buf_idx = np.where(diffs == past.max())[0][0]
            else:
                buf_idx = np.argmin(np.abs(diffs))

        # Get uncertainty at 1/8 resolution
        uncer = self.video.uncertainties[buf_idx]  # [H/8, W/8]

        # Bilinear interpolate raw uncertainty to full resolution first
        uncer_upscaled = F.interpolate(
            uncer.unsqueeze(0).unsqueeze(0),  # [1, 1, H/8, W/8]
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze()  # [img_h, img_w]

        # Then apply linear scaling and inversion
        uncer_rescaled = torch.clamp(45.0 * uncer_upscaled - 35.0, min=0.1)
        mask_fullres = torch.clamp(1.0 / uncer_rescaled, min=0.0, max=1.0)

        # Threshold: pixels below threshold are dynamic
        dynamic_mask = (mask_fullres < threshold).cpu().numpy().astype(np.uint8) * 255

        return dynamic_mask

    def get_uncertainty_masks_for_keyframes(self, img_h, img_w, threshold=0.5):
        """
        Get uncertainty masks for all warmup keyframes.

        Returns:
            list of (frame_idx, mask) tuples where mask is np.ndarray [img_h, img_w] uint8
        """
        kf_num = self.video.counter.value
        kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()
        masks = []
        for i in range(kf_num):
            mask = self.get_uncertainty_mask(int(kf_timestamps[i]), img_h, img_w, threshold)
            masks.append((int(kf_timestamps[i]), mask))
        return masks
