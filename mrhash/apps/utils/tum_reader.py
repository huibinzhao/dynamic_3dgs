from pathlib import Path
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from PIL import Image
import os

class TUMReader:
    def __init__(
        self,
        data_dir: Path,
        min_range=0.01,
        max_range=30,
        depth_scaling=5000.0,
        *args,
        **kwargs,
    ):
        """
        :param data_dir: Directory containing the TUM dataset (should have rgb.txt, depth.txt, and groundtruth.txt or pose.txt)
        :param min_range: minimum range for the points
        :param max_range: maximum range for the points
        :param depth_scaling: scaling factor for depth images (TUM default is 5000)
        """
        self.data_dir = data_dir
        self.min_range = min_range
        self.max_range = max_range
        self.depth_scaling = depth_scaling
        
        # TUM dataset usually has associations. If not, we might need to associate them.
        # For simplicity, we assume we want to read synchronized data or just use what's available.
        # A robust way is to read rgb.txt and depth.txt and associate them by timestamp.
        # But here we will follow the common practice for TUM.
        
        self.rgb_images, self.depth_images, self.poses, self.quats = self.load_tum_data(data_dir)
        self.file_index = 0

    def load_tum_data(self, data_dir: Path):
        rgb_file = data_dir / "rgb.txt"
        depth_file = data_dir / "depth.txt"
        pose_file = data_dir / "groundtruth.txt"
        if not pose_file.exists():
            pose_file = data_dir / "poses.txt"
            
        def read_file_list(filename):
            file_list = []
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    parts = line.split()
                    file_list.append((float(parts[0]), parts[1:]))
            return file_list

        rgb_list = read_file_list(rgb_file)
        depth_list = read_file_list(depth_file)
        pose_list = read_file_list(pose_file)
        
        # Association (nearest neighbor)
        # matches: timestamp, rgb_path, depth_path, tx, ty, tz, qx, qy, qz, qw
        matches = []
        
        # For each rgb, find nearest depth and nearest pose
        depth_timestamps = np.array([item[0] for item in depth_list])
        pose_timestamps = np.array([item[0] for item in pose_list])
        
        for rgb_ts, rgb_data in rgb_list:
            # Find nearest depth
            d_idx = np.argmin(np.abs(depth_timestamps - rgb_ts))
            if np.abs(depth_timestamps[d_idx] - rgb_ts) > 0.02: continue # 20ms threshold
            
            # Find nearest pose
            p_idx = np.argmin(np.abs(pose_timestamps - rgb_ts))
            if np.abs(pose_timestamps[p_idx] - rgb_ts) > 0.02: continue
            
            matches.append({
                'rgb': rgb_data[0],
                'depth': depth_list[d_idx][1][0],
                'pose': pose_list[p_idx][1]
            })
            
        rgb_paths = [data_dir / m['rgb'] for m in matches]
        depth_paths = [data_dir / m['depth'] for m in matches]
        
        poses = []
        quats = []
        for m in matches:
            p = m['pose']
            # TUM pose format: tx ty tz qx qy qz qw
            poses.append(np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32))
            quats.append(np.array([float(p[3]), float(p[4]), float(p[5]), float(p[6])], dtype=np.float32))
            
        return rgb_paths, depth_paths, poses, quats

    def __len__(self):
        return len(self.rgb_images)

    def __iter__(self):
        self.file_index = 0
        return self

    def __next__(self):
        if self.file_index >= len(self):
            raise StopIteration
        result = self[self.file_index]
        self.file_index += 1
        return result

    def __getitem__(self, item) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if item >= len(self):
            raise IndexError("Index out of bounds")

        translation = self.poses[item]
        quat = self.quats[item]

        depth = np.array(Image.open(self.depth_images[item]), dtype=np.float32) / self.depth_scaling
        rgb = np.array(Image.open(self.rgb_images[item]).convert("RGB"), dtype=np.float32)

        return item + 1, translation, quat, depth, rgb
