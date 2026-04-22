"""
DROID-W compatible dataset wrapper for TUM RGB-D data.

This creates a dataset object that DROID-W's tracker can consume,
wrapping around the unified TUM data reader.
"""

import os
import math
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


class DroidWTUMDataset(Dataset):
    """
    A TUM RGB-D dataset class compatible with DROID-W's tracker interface.
    This mimics the interface of DROID-W's TUM_RGBD / BaseDataset class.
    """

    def __init__(self, cfg, color_paths, depth_paths, gt_poses_4x4=None, device="cuda:0"):
        """
        Args:
            cfg: DROID-W compatible config dict
            color_paths: list of str, paths to RGB images
            depth_paths: list of str, paths to depth images
            gt_poses_4x4: optional list/array of [4,4] c2w pose matrices (normalized to first frame)
            device: torch device
        """
        super().__init__()
        self.name = "tumrgbd"
        self.device = device
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]

        self.color_paths = color_paths
        self.depth_paths = depth_paths
        self.n_img = len(color_paths)

        # Camera parameters
        self.H = cfg["cam"]["H"]
        self.W = cfg["cam"]["W"]
        self.fx_orig = cfg["cam"]["fx"]
        self.fy_orig = cfg["cam"]["fy"]
        self.cx_orig = cfg["cam"]["cx"]
        self.cy_orig = cfg["cam"]["cy"]
        self.H_out = cfg["cam"]["H_out"]
        self.W_out = cfg["cam"]["W_out"]
        self.H_edge = cfg["cam"]["H_edge"]
        self.W_edge = cfg["cam"]["W_edge"]

        self.H_out_with_edge = self.H_out + self.H_edge * 2
        self.W_out_with_edge = self.W_out + self.W_edge * 2

        # Compute rescaled intrinsics
        self.intrinsic = torch.as_tensor(
            [self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]
        ).float()
        self.intrinsic[0] *= self.W_out_with_edge / self.W
        self.intrinsic[1] *= self.H_out_with_edge / self.H
        self.intrinsic[2] *= self.W_out_with_edge / self.W
        self.intrinsic[3] *= self.H_out_with_edge / self.H
        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge

        self.fx = self.intrinsic[0].item()
        self.fy = self.intrinsic[1].item()
        self.cx = self.intrinsic[2].item()
        self.cy = self.intrinsic[3].item()

        try:
            from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
            self.fovx = focal2fov(self.fx, self.W_out)
            self.fovy = focal2fov(self.fy, self.H_out)
        except ImportError:
            self.fovx = 2 * math.atan(self.W_out / (2 * self.fx))
            self.fovy = 2 * math.atan(self.H_out / (2 * self.fy))

        self.W_edge_full = int(math.ceil(self.W_edge * self.W / self.W_out_with_edge))
        self.H_edge_full = int(math.ceil(self.H_edge * self.H / self.H_out_with_edge))
        self.H_out_full = self.H - self.H_edge_full * 2
        self.W_out_full = self.W - self.W_edge_full * 2

        self.distortion = None
        self.image_timestamps = None
        self.input_folder = cfg["data"]["input_folder"]

        # GT poses (c2w, normalized to first frame)
        if gt_poses_4x4 is not None:
            self.poses = gt_poses_4x4
        else:
            self.poses = None

        # Store first frame inverse for coordinate alignment
        self.w2c_first_pose = None

    def __len__(self):
        return self.n_img

    def get_color(self, index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)

        color_data = cv2.resize(
            color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge)
        )
        color_data = (
            torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :]
            / 255.0
        )  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]
        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]

        return color_data

    def get_intrinsic(self):
        H_out_with_edge = self.H_out + self.H_edge * 2
        W_out_with_edge = self.W_out + self.W_edge * 2
        intrinsic = torch.as_tensor(
            [self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]
        ).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge
        return intrinsic

    def get_intrinsic_full_resol(self):
        intrinsic = torch.as_tensor(
            [self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]
        ).float()
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge_full
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge_full
        return intrinsic

    def get_color_full_resol(self, index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        color_data_fullsize = (
            torch.from_numpy(color_data_fullsize)
            .float()
            .permute(2, 0, 1)[[2, 1, 0], :, :]
            / 255.0
        )
        color_data_fullsize = color_data_fullsize.unsqueeze(dim=0)

        if self.W_edge_full > 0:
            edge = self.W_edge_full
            color_data_fullsize = color_data_fullsize[:, :, :, edge:-edge]
        if self.H_edge_full > 0:
            edge = self.H_edge_full
            color_data_fullsize = color_data_fullsize[:, :, edge:-edge, :]
        return color_data_fullsize

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif ".npy" in depth_path:
            depth_data = np.load(depth_path).squeeze()
        else:
            raise TypeError(f"Unsupported depth format: {depth_path}")
        depth_data = depth_data.astype(np.float32) / depth_scale
        return depth_data

    def __getitem__(self, index):
        color_data = self.get_color(index)

        depth_data_fullsize = self.depthloader(
            index, self.depth_paths, self.png_depth_scale
        )
        if depth_data_fullsize is not None:
            depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            outsize = (self.H_out_with_edge, self.W_out_with_edge)
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode="nearest"
            )[0, 0]
        else:
            depth_data = torch.zeros(color_data.shape[-2:])

        if self.W_edge > 0:
            edge = self.W_edge
            depth_data = depth_data[:, edge:-edge]
        if self.H_edge > 0:
            edge = self.H_edge
            depth_data = depth_data[edge:-edge, :]

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, pose

    def save_gt_poses(self, path, poses):
        """Save GT poses in TUM format."""
        idx = 0
        with open(path, "w") as f:
            for pose in poses:
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
                translation = pose[:3, 3]
                associated_img_path = self.color_paths[idx]
                timestamp = float(os.path.basename(associated_img_path)[:-4])
                f.write(
                    f"{timestamp} {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} "
                    f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n"
                )
                idx += 1
