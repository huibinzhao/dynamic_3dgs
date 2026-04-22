"""
Unified TUM RGB-D dataloader for integrated DROID-W + Dynamic 3DGS pipeline.

This dataloader reads TUM RGB-D dataset sequences and provides data in
a format usable by both DROID-W (tracking) and dynamic_3dgs (mapping).

It handles:
- Reading and associating rgb.txt, depth.txt, groundtruth.txt
- Providing GT poses (optional)
- Providing data in the format expected by each pipeline component
"""

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


class TUMUnifiedReader:
    """
    Unified TUM RGB-D dataset reader.

    Reads associated RGB images, depth images, and optionally GT poses.
    Provides data interfaces for both DROID-W and dynamic_3dgs.
    """

    def __init__(
        self,
        data_dir: Path,
        min_range: float = 0.1,
        max_range: float = 5.0,
        depth_scaling: float = 5000.0,
        load_gt_pose: bool = True,
    ):
        """
        Args:
            data_dir: Path to TUM dataset directory
            min_range: minimum valid depth (meters)
            max_range: maximum valid depth (meters)
            depth_scaling: depth image scaling factor
            load_gt_pose: whether to load GT poses from groundtruth.txt
        """
        self.data_dir = Path(data_dir)
        self.min_range = min_range
        self.max_range = max_range
        self.depth_scaling = depth_scaling

        # Load and associate data
        (
            self.rgb_paths,
            self.depth_paths,
            self.timestamps,
            self.gt_translations,
            self.gt_quaternions,
            self.gt_poses_4x4,
        ) = self._load_tum_data(self.data_dir, load_gt_pose)

        self._has_gt_pose = load_gt_pose and self.gt_translations is not None
        self.file_index = 0

    @property
    def has_gt_pose(self) -> bool:
        return self._has_gt_pose

    def _load_tum_data(self, data_dir: Path, load_gt_pose: bool):
        """
        Load and associate TUM dataset files.

        Returns:
            rgb_paths: list of Path to RGB images
            depth_paths: list of Path to depth images
            timestamps: list of float timestamps
            gt_translations: list of np.ndarray [3] or None
            gt_quaternions: list of np.ndarray [4] (qx,qy,qz,qw) or None
            gt_poses_4x4: list of np.ndarray [4,4] c2w, normalized to first frame, or None
        """
        rgb_file = data_dir / "rgb.txt"
        depth_file = data_dir / "depth.txt"

        pose_file = data_dir / "groundtruth.txt"
        if not pose_file.exists():
            pose_file = data_dir / "poses.txt"
            if not pose_file.exists():
                load_gt_pose = False

        def read_file_list(filename):
            entries = []
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    entries.append((float(parts[0]), parts[1:]))
            return entries

        rgb_list = read_file_list(rgb_file)
        depth_list = read_file_list(depth_file)
        pose_list = read_file_list(pose_file) if load_gt_pose else None

        # Association by nearest timestamps
        depth_timestamps = np.array([item[0] for item in depth_list])
        pose_timestamps = (
            np.array([item[0] for item in pose_list]) if pose_list else None
        )

        matches = []
        for rgb_ts, rgb_data in rgb_list:
            # Find nearest depth
            d_idx = np.argmin(np.abs(depth_timestamps - rgb_ts))
            if np.abs(depth_timestamps[d_idx] - rgb_ts) > 0.02:
                continue  # 20ms threshold

            match = {
                "rgb": rgb_data[0],
                "depth": depth_list[d_idx][1][0],
                "timestamp": rgb_ts,
            }

            if load_gt_pose and pose_timestamps is not None:
                p_idx = np.argmin(np.abs(pose_timestamps - rgb_ts))
                if np.abs(pose_timestamps[p_idx] - rgb_ts) > 0.02:
                    continue
                match["pose"] = pose_list[p_idx][1]

            matches.append(match)

        rgb_paths = [data_dir / m["rgb"] for m in matches]
        depth_paths = [data_dir / m["depth"] for m in matches]
        timestamps = [m["timestamp"] for m in matches]

        gt_translations = None
        gt_quaternions = None
        gt_poses_4x4 = None

        if load_gt_pose and "pose" in matches[0]:
            gt_translations = []
            gt_quaternions = []
            gt_poses_4x4 = []
            inv_pose = None

            for m in matches:
                p = m["pose"]
                # TUM pose format: tx ty tz qx qy qz qw
                tx, ty, tz = float(p[0]), float(p[1]), float(p[2])
                qx, qy, qz, qw = float(p[3]), float(p[4]), float(p[5]), float(p[6])

                gt_translations.append(
                    np.array([tx, ty, tz], dtype=np.float32)
                )
                gt_quaternions.append(
                    np.array([qx, qy, qz, qw], dtype=np.float32)
                )

                # Build 4x4 c2w matrix (for DROID-W compatibility)
                c2w = np.eye(4)
                c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                c2w[:3, 3] = [tx, ty, tz]

                # Normalize relative to first frame (DROID-W convention)
                if inv_pose is None:
                    inv_pose = np.linalg.inv(c2w)
                    gt_poses_4x4.append(np.eye(4))
                else:
                    gt_poses_4x4.append(inv_pose @ c2w)

        return (
            rgb_paths,
            depth_paths,
            timestamps,
            gt_translations,
            gt_quaternions,
            gt_poses_4x4,
        )

    def __len__(self):
        return len(self.rgb_paths)

    def __iter__(self):
        self.file_index = 0
        return self

    def __next__(self):
        if self.file_index >= len(self):
            raise StopIteration
        result = self[self.file_index]
        self.file_index += 1
        return result

    def __getitem__(
        self, item
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a frame in dynamic_3dgs format.

        Returns:
            frame: 1-based frame index
            translation: np.ndarray [3] (GT translation, or zeros if no GT)
            quaternion: np.ndarray [4] (qx, qy, qz, qw) (GT or zeros)
            depth: np.ndarray [H, W] in meters
            rgb: np.ndarray [H, W, 3] as float32
        """
        if item >= len(self):
            raise IndexError("Index out of bounds")

        depth = (
            np.array(Image.open(self.depth_paths[item]), dtype=np.float32)
            / self.depth_scaling
        )
        rgb = np.array(
            Image.open(self.rgb_paths[item]).convert("RGB"), dtype=np.float32
        )

        if self._has_gt_pose:
            translation = self.gt_translations[item]
            quaternion = self.gt_quaternions[item]
        else:
            translation = np.zeros(3, dtype=np.float32)
            quaternion = np.array([0, 0, 0, 1], dtype=np.float32)

        return item + 1, translation, quaternion, depth, rgb

    def get_color_paths(self) -> List[str]:
        """Get list of color image paths as strings."""
        return [str(p) for p in self.rgb_paths]

    def get_depth_paths(self) -> List[str]:
        """Get list of depth image paths as strings."""
        return [str(p) for p in self.depth_paths]

    def get_gt_poses_4x4(self) -> Optional[List[np.ndarray]]:
        """
        Get GT poses as 4x4 c2w matrices, normalized to first frame.
        Returns None if no GT poses available.
        """
        return self.gt_poses_4x4

    def get_frame_with_custom_pose(
        self, item: int, translation: np.ndarray, quaternion: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a frame with a custom (estimated) pose instead of GT.

        Args:
            item: frame index (0-based)
            translation: np.ndarray [3]
            quaternion: np.ndarray [4] (qx, qy, qz, qw)

        Returns:
            Same format as __getitem__
        """
        if item >= len(self):
            raise IndexError("Index out of bounds")

        depth = (
            np.array(Image.open(self.depth_paths[item]), dtype=np.float32)
            / self.depth_scaling
        )
        rgb = np.array(
            Image.open(self.rgb_paths[item]).convert("RGB"), dtype=np.float32
        )

        return item + 1, translation, quaternion, depth, rgb
