from __future__ import annotations
import dataclasses
import functools
import os
import time

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast, Dict, DefaultDict
from collections import defaultdict, deque
import viser.transforms as vtf
import viser
import torch
import torchvision
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from cv_bridge import CvBridge  # Needed for converting between ROS Image messages and OpenCV images
from torch.nn import Parameter
import cv2
from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper

from nerfstudio.cameras.camera_utils import get_distortion_params
from nerfstudio.cameras.cameras import Cameras,CameraType
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.process_data.colmap_utils import qvec2rotmat
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils import profiler, writer
from copy import deepcopy
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.viewer.viewer_elements import ViewerButton, ViewerCheckbox
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState

import nerfstudio.utils.poses as pose_utils
import numpy as np
import scipy.spatial.transform as transform
import rclpy
from rclpy.node import Node
from lifelong_msgs.msg import ImagePose
from lifelong_msgs.msg import ImagePoses
from l3gs.L3GS_pipeline import L3GSPipeline
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from l3gs.L3GS_utils import Utils as U

TORCH_DEVICE = str
TRAIN_ITERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]

def ros_pose_to_nerfstudio(pose_msg: Pose, static_transform=None):
    """
    Takes a ROS Pose message and converts it to the
    3x4 transform format used by nerfstudio.
    """
    quat = np.array(
        [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ],
    )
    posi = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    R = torch.tensor(qvec2rotmat(quat))
    T = torch.cat([R, posi.unsqueeze(-1)], dim=-1)
    T = T.to(dtype=torch.float32)
    if static_transform is not None:
        T = pose_utils.multiply(T, static_transform)
        T2 = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("x", 90, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("z", 180, degrees=True).as_matrix()
        R3 = transform.Rotation.from_euler("y", 180, degrees=True).as_matrix()
        R = torch.from_numpy(R3 @ R2 @ R1)
        T2[:, :3] = R
        T = pose_utils.multiply(T2, T)


    return T.to(dtype=torch.float32)

def pop_n_elements(deque_obj, n):
    popped_elements = []
    for _ in range(min(n, len(deque_obj))):  # Ensure you don't pop more elements than exist
        popped_elements.append(deque_obj.popleft())
    return popped_elements

def inverse_SE3(A):
    # A is expected to be a 4x4 SE(3) matrix
    R = A[:3, :3]  # Extract the rotation part
    t = A[:3, 3]   # Extract the translation part

    R_inv = R.t()  # Transpose of R for the inverse rotation
    t_inv = -torch.matmul(R_inv, t)  # Apply inverse rotation to the negated translation

    # Construct the inverse SE(3) matrix
    A_inv = torch.eye(4)  # Start with an identity matrix
    A_inv[:3, :3] = R_inv
    A_inv[:3, 3] = t_inv

    return A_inv


@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients over."""

class TrainerNode(Node):
    def __init__(self,trainer):
        super().__init__('trainer_node')
        self.trainer_ = trainer
        self.subscription_ = self.create_subscription(ImagePose,"/camera/color/imagepose",self.add_img_callback,100)

    def add_img_callback(self,msg):
        print("Appending imagepose to queue",flush=True)
        self.trainer_.image_add_callback_queue.append(msg)

class TricamTrainerNode(Node):
    def __init__(self,trainer):
        super().__init__('trainer_node')
        self.trainer_ = trainer
        self.subscription_ = self.create_subscription(ImagePoses,"/camera/color/imagepose",self.add_img_callback,100)

    def add_img_callback(self,msg):
        print("Appending imagepose to queue",flush=True)
        # self.trainer_.image_add_callback_queue.append(msg)
        if msg.got_prev_poses is False:
            self.trainer_.image_add_callback_queue.append((msg.image_poses[0], msg.points, msg.colors, None, msg.got_prev_poses))
        else:
            self.trainer_.image_add_callback_queue.append((None, None, None, msg.prev_poses, msg.got_prev_poses))

        # self.trainer_.image_add_callback_queue.append((msg.image_poses[1], None))

        # self.trainer_.image_add_callback_queue.append(msg.image_poses[2])

class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: L3GSPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: TORCH_DEVICE = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.training_state: Literal["training", "paused", "completed"] = "training"
        self.gas_int: int = 1
        self.gradient_accumulation_steps: DefaultDict = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps)

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)

        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None
        self.image_add_callback_queue = []
        self.add_to_clip_queue = []
        self.image_process_queue = []
        self.query_diff_queue = []
        self.query_diff_size = 5*3
        self.query_diff_msg_queue = []
        self.query_diff_size = 1
        self.cvbridge = CvBridge()
        self.clip_out_queue = mp.Queue()
        self.dino_out_queue = mp.Queue()
        self.done_scale_calc = False
        self.calculate_diff = False # whether or not to calculate the image diff
        self.calulate_metrics = False # whether or not to calculate the metrics
        self.num_boxes_added = 0
        self.box_viser_handles = []
        self.deprojected_queue = deque()
        self.colors_queue = deque()
        self.points_queue = deque()
        self.start_idx = 0
        self.train_lerf = False
        self.diff_wait_counter = 0
        self.multicam = True

    def handle_stage_btn(self, handle: ViewerButton):
        import os.path as osp
        bag_path = osp.join(
            # '/home/lerf/lifelong-lerf/experiment_bags',
            '/home/lerf/lifelong-lerf/bag',
            self.pipeline.lifelong_exp_aname.value,
            f'loop{self.pipeline.lifelong_exp_loop.value + 1}'
        )
        # check if the bag exists
        if not osp.exists(bag_path):
            print("Bag not found at", bag_path)
            return
        self.pipeline.lifelong_exp_aname.set_disabled(True) #dropdown disabling doesn't seem to work?
        
        if self.pipeline.lifelong_exp_loop.value > 0:
            self.pipeline.datamanager.train_dataset.stage.append(len(self.pipeline.datamanager.train_dataset))
            self.calculate_diff = True
            print("Stage set to", self.pipeline.datamanager.train_dataset.stage[-1])
    
        self.query_diff_msg_queue = []
        self.query_diff_queue = []

        # # keeping just in case...
        # if self.pipeline.lifelong_exp_loop.value > 0: # i.e., exp_loop = 2 means that youve already played loop 1 and 2.
        #     stage = self.pipeline.datamanager.train_dataset.stage
        #     mask = self.pipeline.datamanager.train_dataset.mask_tensor[stage[-2]:stage[-1], ...] #Mask is NxHxWx1
        #     mask_sum = torch.sum(mask)
        #     # Calculate the percentage of masked pixels
        #     percentage_masked = 1 - (mask_sum / (mask.shape[0] * mask.shape[1] * mask.shape[2]))
        #     # Save the percentage to a file with name of the stage
        #     with open(f"metrics/metrics_{self.pipeline.lifelong_exp_aname.value}_{self.pipeline.lifelong_exp_loop.value}.txt", "w") as f:
        #         f.write(f"Precentage Masked: {percentage_masked}\n")   
        self.pipeline.stage_button.set_disabled(True)
        self.pipeline.lifelong_exp_loop.value += 1

        self.num_boxes_added = 0 # reset the mask boxes of this scene
        for bbox_viser in self.box_viser_handles:
            bbox_viser.remove()

        # start the bag
        # self.pipeline.lifelong_exp_start.set_disabled(True)
        import subprocess
        proc = subprocess.Popen(['ros2', 'bag', 'play', bag_path, "--rate", "1.0","--topics", "/camera/color/camera_info", "/sim_realsense"])
        print("Started bag at", bag_path)
        proc.communicate()
        proc.terminate()
        print("Terminated bag at", bag_path)
        self.pipeline.stage_button.set_disabled(False)
        
    def handle_calc_metric(self, handle: ViewerButton):
        self.pipeline.calc_metric.disabled = True
        self.calulate_metrics = True

    def handle_percentage_masked(self, handle: ViewerButton):
        # Calculate the sum of masked pixels
        stage = self.pipeline.datamanager.train_dataset.stage
        mask = self.pipeline.datamanager.train_dataset.mask_tensor[stage[-2]:stage[-1], ...] #Mask is NxHxWx1
        mask_sum = torch.sum(mask)
        # Calculate the percentage of masked pixels
        percentage_masked = 1 - (mask_sum / (mask.shape[0] * mask.shape[1] * mask.shape[2]))
        # Save the percentage to a file with name of the stage
        with open(f"metrics/metrics_{self.pipeline.lifelong_exp_aname.value}_{self.pipeline.lifelong_exp_loop.value}.txt", "w") as f:
            f.write(f"Precentage Masked: {percentage_masked}\n")

    def calc_metric(self):
        self.pipeline.model.eval()
        from tqdm import tqdm
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        import pdb; pdb.set_trace()
        cameras = self.pipeline.datamanager.train_dataset.cameras[:len(self.pipeline.datamanager.train_dataset)]
        stage = self.pipeline.datamanager.train_dataset.stage[-1]
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        import matplotlib.pyplot as plt
        for i in tqdm(range(stage, len(self.pipeline.datamanager.train_dataset))):
            cam = cameras[i].to(self.device)
            gt_rgb = self.pipeline.datamanager.train_dataset.image_tensor[i].to(self.device)
            gt_rgb = gt_rgb.permute(2,0,1).unsqueeze(0)
            with torch.no_grad():
                ray_bundle = cam.generate_rays(0)
                ray_bundle.metadata['rgb_only'] = torch.ones((ray_bundle.origins.shape[0],ray_bundle.origins.shape[1], 1), dtype=torch.bool, device=ray_bundle.origins.device)
                predicted_rgb = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)['rgb'].to(self.device)
            predicted_rgb = predicted_rgb.permute(2,0,1).unsqueeze(0)
            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(gt_rgb[0].permute(1,2,0).cpu().numpy())
            axes[0].set_title('GT image')
            axes[1].imshow(predicted_rgb[0].permute(1,2,0).cpu().numpy())
            axes[1].set_title(f'Predicted image, PSNR={psnr}')
            plt.show()
            avg_psnr += psnr
            avg_ssim += ssim
            avg_lpips += lpips
        avg_psnr /= (len(cameras) - stage)
        avg_ssim /= (len(cameras) - stage)
        avg_lpips /= (len(cameras) - stage)
        import pdb; pdb.set_trace()

        #Save the metrics to a file with name of the stage
        with open(f"metrics/metrics_{self.pipeline.lifelong_exp_aname.value}_{self.pipeline.lifelong_exp_loop.value}.txt", "w") as f:
            f.write(f"PSNR: {avg_psnr}\n")
            f.write(f"SSIM: {avg_ssim}\n")
            f.write(f"LPIPS: {avg_lpips}\n")
        self.pipeline.calc_metric.disabled = False
        self.calulate_metrics = False
        self.pipeline.model.train()


    def handle_start_droidslam(self, handle: ViewerButton):
        import subprocess
        with subprocess.Popen(['env', 'CUDA_VISIBLE_DEVICES=0', 'python', 'ros_node.py'], cwd='/home/lerf/DROID-SLAM') as proc:
            print("Started droidslam")
            self.pipeline.droidslam_start.set_disabled(True)
            self.pipeline.stage_button.set_disabled(False)
            proc.communicate()
        print("Terminated droidslam")
        # proc.terminate()

    def handle_train_lerf(self, handle: ViewerButton):
        print("Training LERF")
        self.train_lerf = True

    def handle_diff(self, handle: ViewerButton):
        print("Diff")
        self.calculate_diff = True

    def add_img_callback(self, msg:ImagePose, decode_only=False):
        '''
        this function queues things to be added
        returns the image, depth, and pose if the dataparser is defined yet, otherwise None
        if decode_only, don't add the image to the clip/dino queue using `add_image`.
        '''
        # image: Image = msg.img
        # camera_to_worlds = ros_pose_to_nerfstudio(msg.pose)
        # print('self.imgidx: ' + str(self.imgidx))
        # # CONSOLE.print("Adding image to dataset")
        # # image_data = torch.tensor(image.data, dtype=torch.uint8).view(image.height, image.width, -1).to(torch.float32)/255.
        image_data = torch.tensor(self.cvbridge.compressed_imgmsg_to_cv2(msg.img, 'rgb8'),dtype = torch.float32)/255.
        
        # fx = torch.tensor([msg.fl_x])
        # fy = torch.tensor([msg.fl_y])
        # cy = torch.tensor([msg.cy])
        # cx = torch.tensor([msg.cx])
        # cx = torch.tensor([msg.cy])
        # cy = torch.tensor([msg.cx])

        ### Multicamera Support
        # width = torch.tensor([msg.w])
        # height = torch.tensor([msg.h])
        # distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        # camera_type = CameraType.PERSPECTIVE
        # camera = Cameras(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type).reshape(())
        # K = camera.get_intrinsics_matrices().numpy()

        # if self.multicam:
        #     crop_top = 60
        #     crop_bottom = 480 - 60
        #     if msg.w != msg.img.width or msg.h != msg.img.height:
        #         # crop image_data to new_W new_H centered
        #         image_data = image_data[crop_top:crop_bottom, :, :].permute(2, 0, 1).unsqueeze(0)
        #         # print('after croppping', image_data.shape)
        #         import torch.nn.functional as F
        #         image_data = F.interpolate(image_data, size=(msg.img.height, msg.img.width), mode='bilinear', align_corners=False)
        #         # back to HxWxC
        #         image_data = image_data.permute(2, 3, 1, 0)[:,:,:,0]
        #         # print('after resize', image_data.shape)
        # K, image_data, mask = self._undistort_image(camera, get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3).numpy(), {}, image_data.cpu().numpy(), K)
        # image_data = torch.from_numpy(image_data).to(torch.float32)

        if not decode_only:
            # with self.train_lock:
            self.pipeline.add_image(img = image_data)
        # dep_out *= self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        # #TODO add the dataparser transform here
        # H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        # row = torch.tensor([[0,0,0,1]],dtype=torch.float32,device=retc.camera_to_worlds.device)
        # retc.camera_to_worlds = torch.matmul(torch.cat([H,row]),torch.cat([retc.camera_to_worlds,row]))[:3,:]
        # retc.camera_to_worlds[:3,3] *= self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        # if not self.done_scale_calc:
        #     return None,None,None
        # return img_out, dep_out, retc

    # @profile
    def _undistort_image(self, camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
        mask = None
        if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
            distortion_params = np.array(
                [
                    distortion_params[0],
                    distortion_params[1],
                    distortion_params[4],
                    distortion_params[5],
                    distortion_params[2],
                    distortion_params[3],
                    0,
                    0,
                ]
            )
            if np.any(distortion_params):
                newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
            else:
                newK = K
                roi = 0, 0, image.shape[1], image.shape[0]
            # crop the image and update the intrinsics accordingly
            x, y, w, h = roi
            image = image[y : y + h, x : x + w]
            if "depth_image" in data:
                data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
            if "mask" in data:
                mask = data["mask"].numpy()
                mask = mask.astype(np.uint8) * 255
                if np.any(distortion_params):
                    mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
                mask = mask[y : y + h, x : x + w]
                mask = torch.from_numpy(mask).bool()
            K = newK

        elif camera.camera_type.item() == CameraType.FISHEYE.value:
            distortion_params = np.array(
                [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
            )
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
            )
            # and then remap:
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            if "mask" in data:
                mask = data["mask"].numpy()
                mask = mask.astype(np.uint8) * 255
                mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
                mask = torch.from_numpy(mask).bool()
            K = newK
        elif camera.camera_type.item() == CameraType.FISHEYE624.value:
            fisheye624_params = torch.cat(
                [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
            )
            assert fisheye624_params.shape == (16,)
            assert (
                "mask" not in data
                and camera.metadata is not None
                and "fisheye_crop_radius" in camera.metadata
                and isinstance(camera.metadata["fisheye_crop_radius"], float)
            )
            fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

            # Approximate the FOV of the unmasked region of the camera.
            upper, lower, left, right = fisheye624_unproject_helper(
                torch.tensor(
                    [
                        [camera.cx, camera.cy - fisheye_crop_radius],
                        [camera.cx, camera.cy + fisheye_crop_radius],
                        [camera.cx - fisheye_crop_radius, camera.cy],
                        [camera.cx + fisheye_crop_radius, camera.cy],
                    ],
                    dtype=torch.float32,
                )[None],
                params=fisheye624_params[None],
            ).squeeze(dim=0)
            fov_radians = torch.max(
                torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
                torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
            )

            # Heuristics to determine parameters of an undistorted image.
            undist_h = int(fisheye_crop_radius * 2)
            undist_w = int(fisheye_crop_radius * 2)
            undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
            undist_K = torch.eye(3)
            undist_K[0, 0] = undistort_focal  # fx
            undist_K[1, 1] = undistort_focal  # fy
            undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
            undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

            # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
            undist_uv_homog = torch.stack(
                [
                    *torch.meshgrid(
                        torch.arange(undist_w, dtype=torch.float32),
                        torch.arange(undist_h, dtype=torch.float32),
                    ),
                    torch.ones((undist_w, undist_h), dtype=torch.float32),
                ],
                dim=-1,
            )
            assert undist_uv_homog.shape == (undist_w, undist_h, 3)
            dist_uv = (
                fisheye624_project(
                    xyz=(
                        torch.einsum(
                            "ij,bj->bi",
                            torch.linalg.inv(undist_K),
                            undist_uv_homog.reshape((undist_w * undist_h, 3)),
                        )[None]
                    ),
                    params=fisheye624_params[None, :],
                )
                .reshape((undist_w, undist_h, 2))
                .numpy()
            )
            map1 = dist_uv[..., 1]
            map2 = dist_uv[..., 0]

            # Use correspondence to undistort image.
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

            # Compute undistorted mask as well.
            dist_h = camera.height.item()
            dist_w = camera.width.item()
            mask = np.mgrid[:dist_h, :dist_w]
            mask[0, ...] -= dist_h // 2
            mask[1, ...] -= dist_w // 2
            mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
            mask = torch.from_numpy(
                cv2.remap(
                    mask.astype(np.uint8) * 255,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                / 255.0
            ).bool()[..., None]
            assert mask.shape == (undist_h, undist_w, 1)
            K = undist_K.numpy()
        else:
            raise NotImplementedError("Only perspective and fisheye cameras are supported")

        return K, image, mask


    def process_query_diff(self, msg:ImagePose, step, clip_dict = None, dino_data = None):
        self.diff_wait_counter -= 1
        if self.diff_wait_counter > 0:
            return

        heatmaps, heatmap_masks, gsplat_outputs_list, poses, depths, images = [], [], [], [], [], []

        c2w = ros_pose_to_nerfstudio(msg.pose)
        H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        row = torch.tensor([[0,0,0,1]],dtype=torch.float32,device=c2w.device)
        c2w= torch.matmul(torch.cat([H,row]),torch.cat([c2w,row]))[:3,:]
        c2w[:3,3] *= self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale

        image = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.img,'rgb8'),dtype = torch.float32)/255.
        depth = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.depth,'16UC1') / 1000. ,dtype = torch.float32)
        image, depth = image.to(self.device), depth.to(self.device)
        fx = torch.tensor([msg.fl_x])
        fy = torch.tensor([msg.fl_y])
        cy = torch.tensor([msg.cy])
        cx = torch.tensor([msg.cx])
        # cx = torch.tensor([msg.cy])
        # cy = torch.tensor([msg.cx])
        width = torch.tensor([msg.w])
        height = torch.tensor([msg.h])
        distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        camera_type = CameraType.PERSPECTIVE
        pose = Cameras(c2w, fx, fy, cx, cy, width, height, distortion_params, camera_type)
        pose.camera_type = pose.camera_type.unsqueeze(0)

        heat_map, cleaned_heatmap_mask, gsplat_outputs = self.pipeline.query_diff(image, pose, depth, step, vis_verbose = self.pipeline.plot_verbose)
        # import pdb; pdb.set_trace()
        # if self.pipeline.use_clip:
        #     heat_map_mask = heat_map > -0.85
        # else:
        #     heat_map_mask = heat_map

        heatmaps.append(heat_map)
        heatmap_masks.append(cleaned_heatmap_mask)
        gsplat_outputs_list.append(gsplat_outputs)
        images.append(image)
        poses.append(pose)
        depths.append(depth)
        images.append(image)

        # affected_gaussians_idxs = self.pipeline.heatmaps2gaussians(heatmap_masks, gsplat_outputs_list, poses, depths, images)

        boxes, points_tr = self.pipeline.heatmaps2box(heatmaps, heatmap_masks, images, poses, depths, depths, gsplat_outputs_list)

        gaussian_means_ptcld = self.pipeline.model.means.detach().cpu().numpy()
        colors = np.ones_like(gaussian_means_ptcld)
        self.viewer_state.viser_server.add_point_cloud(
            '/means_ptcld',
            points=gaussian_means_ptcld * 10,
            colors=colors,
            point_size=0.3,
        )

        if len(boxes) > 0:
            # Change detected!
            self.viewer_state.viser_server.add_point_cloud(
                '/pointcloud',
                points=points_tr.vertices * 10,
                colors=points_tr.visual.vertex_colors[:, :3],
            )

            for obox_ind, obox in enumerate(boxes):
                import trimesh
                bbox = trimesh.creation.box(
                    extents=obox.S.cpu().numpy() * 10
                )
                bbox_viser = self.viewer_state.viser_server.add_mesh_trimesh(
                    f"/bbox_{obox_ind}",
                    bbox,
                    # scale=self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
                    wxyz=vtf.SO3.from_matrix(obox.R.cpu().numpy()).wxyz,
                    position=obox.T.cpu().numpy() * 10,
                )
                print("just visualized the box.")

                affected_gaussians_idxs = self.pipeline.bbox2gaussians(obox)
                inside = obox.within(self.pipeline.model.means)
                # print(affected_gaussians_idxs.shape, self.pipeline.model.means.shape)
                # self.pipeline.model.means = Parameter(self.pipeline.model.means[~affected_gaussians_idxs].detach())
            self.diff_wait_counter = 5

            print(poses, affected_gaussians_idxs)
        # TODO: mask out regions in all training images
    
    def deproject_to_RGB_point_cloud(self, image, depth_image, camera, num_samples = 800, device = 'cuda:0'):
        """
        Converts a depth image into a point cloud in world space using a Camera object.
        """
        scale = self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        # import pdb; pdb.set_trace()
        H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        # c2w = camera.camera_to_worlds.cpu()
        # depth_image = depth_image.cpu()
        # image = image.cpu()
        c2w = camera.camera_to_worlds.to(device)
        depth_image = depth_image.to(device)
        image = image.to(device)
        fx = camera.fx.item()
        fy = camera.fy.item()
        # cx = camera.cx.item()
        # cy = camera.cy.item()

        _, _, height, width = depth_image.shape

        grid_x, grid_y = torch.meshgrid(torch.arange(width, device = device), torch.arange(height, device = device), indexing='ij')
        grid_x = grid_x.transpose(0,1).float()
        grid_y = grid_y.transpose(0,1).float()

        flat_grid_x = grid_x.reshape(-1)
        flat_grid_y = grid_y.reshape(-1)
        flat_depth = depth_image[0, 0].reshape(-1)
        flat_image = image.reshape(-1, 3)

        ### simple uniform sampling approach
        # num_points = flat_depth.shape[0]
        # sampled_indices = torch.randint(0, num_points, (num_samples,))
        non_zero_depth_indices = torch.nonzero(flat_depth != 0).squeeze()

        # Ensure there are enough non-zero depth indices to sample from
        if non_zero_depth_indices.numel() < num_samples:
            num_samples = non_zero_depth_indices.numel()
        # Sample from non-zero depth indices
        sampled_indices = non_zero_depth_indices[torch.randint(0, non_zero_depth_indices.shape[0], (num_samples,))]

        sampled_depth = flat_depth[sampled_indices] * scale
        # sampled_depth = flat_depth[sampled_indices]
        sampled_grid_x = flat_grid_x[sampled_indices]
        sampled_grid_y = flat_grid_y[sampled_indices]
        sampled_image = flat_image[sampled_indices]

        X_camera = (sampled_grid_x - width/2) * sampled_depth / fx
        Y_camera = -(sampled_grid_y - height/2) * sampled_depth / fy

        ones = torch.ones_like(sampled_depth)
        P_camera = torch.stack([X_camera, Y_camera, -sampled_depth, ones], dim=1)
        
        homogenizing_row = torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype, device=device)
        camera_to_world_homogenized = torch.cat((c2w, homogenizing_row), dim=0)

        P_world = torch.matmul(camera_to_world_homogenized, P_camera.T).T
        
        return P_world[:, :3], sampled_image
    
    def deproject_droidslam_point_cloud(self, colors, points, frame1, num_samples = 500, device = 'cuda:0'):
        """
        Converts a depth image into a point cloud in world space using a Camera object.
        """
        num_3d_pts = len(points)//3

        scale = self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        frame1 = frame1.camera_to_worlds.to(device)

        # RANDOM SAMPLING
        sampled_indices = torch.randint(0, num_3d_pts, (num_samples,))

        points = torch.reshape(torch.Tensor(points).to(device), (num_3d_pts, 3))
        points[:, 2] = -points[:, 2]
        points[:, 1] = -points[:, 1]
        P_world = points[sampled_indices] * scale

        P_world = torch.cat([P_world, torch.ones((P_world.shape[0], 1), device=device)], dim=1)
        homogenizing_row = torch.tensor([[0, 0, 0, 1]], dtype=frame1.dtype, device=device)
        frame1_homogenized = torch.cat((frame1, homogenizing_row), dim=0)

        P_world = torch.matmul(frame1_homogenized, P_world.T).T
        
        colors = torch.reshape(torch.Tensor(colors).to(device), (num_3d_pts, 3))
        return P_world[:, :3], colors[sampled_indices]
    
    # @profile
    def process_image(self, msg:ImagePose, step, points, clrs, clip_dict = None, dino_data = None):
        '''
        This function actually adds things to the dataset
        '''
        # start = time.time()
        camera_to_worlds = ros_pose_to_nerfstudio(msg.pose)
        # CONSOLE.print("Adding image to dataset")
        image_data = torch.tensor(self.cvbridge.compressed_imgmsg_to_cv2(msg.img, 'rgb8'),dtype = torch.float32)/255.
        depth = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.depth,'16UC1').astype(np.int16),dtype = torch.int16)/1000.
        # import pdb; pdb.set_trace()
        fx = torch.tensor([msg.fl_x])
        fy = torch.tensor([msg.fl_y])
        cy = torch.tensor([msg.cy])
        cx = torch.tensor([msg.cx])
        # print('fx', fx, 'fy', fy, 'cx', cx, 'cy', cy)
        # import pdb; pdb.set_trace()

        # cx = torch.tensor([msg.cy])
        # cy = torch.tensor([msg.cx])
        # width = torch.tensor([msg.w])
        # height = torch.tensor([msg.h])
        # distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        # camera_type = CameraType.PERSPECTIVE
        # c = Cameras(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type)

        ### Multicamera Support
        width = torch.tensor([msg.w])
        height = torch.tensor([msg.h])
        distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        camera_type = CameraType.PERSPECTIVE
        camera = Cameras(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type).reshape(())
        # K = camera.get_intrinsics_matrices().numpy()

        # if self.multicam:
        #     crop_top = 60
        #     crop_bottom = 480 - 60
        #     if msg.w != msg.img.width or msg.h != msg.img.height:
        #         # crop image_data to new_W new_H centered
        #         image_data = image_data[crop_top:crop_bottom, :, :].permute(2, 0, 1).unsqueeze(0)
        #         # print('after croppping', image_data.shape)
        #         import torch.nn.functional as F
        #         image_data = F.interpolate(image_data, size=(msg.img.height, msg.img.width), mode='bilinear', align_corners=False)
        #         # back to HxWxC
        #         image_data = image_data.permute(2, 3, 1, 0)[:,:,:,0]
        #         # print('after resize', image_data.shape)
        # K, image_data, mask = self._undistort_image(camera, get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3).numpy(), {}, image_data.cpu().numpy(), K)
        # image_data = torch.from_numpy(image_data).to(torch.float32)
        # fx = float(K[0, 0])
        # fy = float(K[1, 1])
        # cx = float(K[0, 2])
        # cy = float(K[1, 2])
        # width = image_data.shape[1]
        # height = image_data.shape[0]
        # distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        # camera_type = CameraType.PERSPECTIVE
        # camera = Cameras(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type)

        with self.train_lock:
            self.pipeline.process_image(img = image_data, depth = depth, pose = camera, clip=clip_dict, dino=dino_data)
        # print("Done processing image")
        image_uint8 = (image_data * 255).detach().type(torch.uint8)
        image_uint8 = image_uint8.permute(2, 0, 1)
        image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100)  # type: ignore
        image_uint8 = image_uint8.permute(1, 2, 0)
        image_uint8 = image_uint8.cpu().numpy()
        idx = len(self.pipeline.datamanager.train_dataset)-1
        dataset_cam = self.pipeline.datamanager.train_dataset.cameras[idx]
        # print(dataset_cam)
        c2w = dataset_cam.camera_to_worlds.cpu().numpy()
        R = vtf.SO3.from_matrix(c2w[:3, :3])
        R = R @ vtf.SO3.from_x_radians(np.pi)
        cidx = self.viewer_state._pick_drawn_image_idxs(idx+1)[-1]
        # scale_factor = self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        # camera_handle = self.viser_server.add_camera_frustum(
        #         name=f"/cameras/camera_{idx:05d}",
        #         fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
        #         scale=self.config.camera_frustum_scale,
        #         aspect=float(camera.cx[0] / camera.cy[0]),
        #         image=image_uint8,
        #         wxyz=R.wxyz,
        #         position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
        #     )
        # scale = self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        camera_handle = self.viewer_state.viser_server.add_camera_frustum(
                    name=f"/cameras/camera_{cidx:05d}",
                    fov=2 * np.arctan(float(dataset_cam.cx / dataset_cam.fx[0])),
                    scale= 0.5,
                    aspect=float(dataset_cam.cx[0] / dataset_cam.cy[0]),
                    image=image_uint8,
                    wxyz=R.wxyz,
                    position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO # SCALE,
                )
        
        @camera_handle.on_click
        def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
            with event.client.atomic():
                event.client.camera.position = event.target.position
                event.client.camera.wxyz = event.target.wxyz
        self.viewer_state.camera_handles[cidx] = camera_handle
        self.viewer_state.original_c2w[cidx] = c2w

        if not self.multicam:
            project_interval = 4
            if self.done_scale_calc and msg.depth is not None and idx % project_interval == 0:
                depth = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.depth,'16UC1').astype(np.int16),dtype = torch.int16)/1000.
                depth = depth.unsqueeze(0).unsqueeze(0)
                if depth.shape[2] != image_data.shape[0] or depth.shape[3] != image_data.shape[1]:
                    import torch.nn.functional as F
                    depth = F.interpolate(depth, size=(image_data.shape[0], image_data.shape[1]), mode='bilinear', align_corners=False)
                deprojected, colors = self.deproject_to_RGB_point_cloud(image_data, depth, dataset_cam)
                self.deprojected_queue.extend(deprojected)
                self.colors_queue.extend(colors)

            elif self.done_scale_calc and msg.depth is None and idx % project_interval == 0:
                depth = self.pipeline.monodepth_inference(image_data.numpy())
                # depth = torch.rand((1,1,480,640))
                deprojected, colors = self.deproject_to_RGB_point_cloud(image_data, depth, dataset_cam)
                self.deprojected_queue.extend(deprojected)
                self.colors_queue.extend(colors)
        else:
            # REALSENSE DEPTH
            # rs_interval = 3
            # if self.done_scale_calc and msg.depth.encoding != '' and idx % rs_interval == 0:
            #     # depth = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.depth,'16UC1').astype(np.int16),dtype = torch.int16)/1000.
            #     depth = depth.unsqueeze(0).unsqueeze(0)
            #     if depth.shape[2] != image_data.shape[0] or depth.shape[3] != image_data.shape[1]:
            #         import torch.nn.functional as F
            #         depth = F.interpolate(depth, size=(image_data.shape[0], image_data.shape[1]), mode='bilinear', align_corners=False)
            #     deprojected, colors = self.deproject_to_RGB_point_cloud(image_data, depth, dataset_cam)
            #     self.deprojected_queue.extend(deprojected)
            #     self.colors_queue.extend(colors)
            # DROIDSLAM
            if self.done_scale_calc and points:
                # import pdb; pdb.set_trace()
                # self.points_queue.extend(points)
                frame1 = self.pipeline.datamanager.train_dataset.cameras[0]
                deprojected, colors = self.deproject_droidslam_point_cloud(clrs, points, frame1)
                self.deprojected_queue.append(deprojected)
                self.colors_queue.append(colors)
            # else:
            #     project_interval = 3
            #     rs_interval = 3
            #     if self.done_scale_calc and msg.depth.encoding != '' and idx % rs_interval == 0:
            #         depth = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.depth,'16UC1').astype(np.int16),dtype = torch.int16)/1000.
            #         depth = depth.unsqueeze(0).unsqueeze(0)
            #         if depth.shape[2] != image_data.shape[0] or depth.shape[3] != image_data.shape[1]:
            #             import torch.nn.functional as F
            #             depth = F.interpolate(depth, size=(image_data.shape[0], image_data.shape[1]), mode='bilinear', align_corners=False)
            #         deprojected, colors = self.deproject_to_RGB_point_cloud(image_data, depth, dataset_cam)
            #         self.deprojected_queue.extend(deprojected)
            #         self.colors_queue.extend(colors)
            #     elif self.done_scale_calc and msg.depth.encoding == '' and idx % project_interval == 0:
            #         depth = self.pipeline.monodepth_inference(image_data.numpy())
            #         # depth = torch.rand((1,1,480,640))
            #         deprojected, colors = self.deproject_to_RGB_point_cloud(image_data, depth, dataset_cam) #, num_samples = 40)
            #         self.deprojected_queue.extend(deprojected)
            #         self.colors_queue.extend(colors)

    # def add_to_clip(clip_dict = None):
    #     self.pipeline.add_to_clip

    def update_poses(self, BA_deltas, start_idx):
        idxs = list(self.viewer_state.camera_handles.keys())
        missed_depro = len(BA_deltas) - len(self.deprojected_queue)
        for idx in range(start_idx, len(self.pipeline.datamanager.train_dataset)):

            C = self.pipeline.datamanager.train_dataset.cameras

            c2w = C.camera_to_worlds[idx,...]

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            self.viewer_state.camera_handles[idxs[idx]].position = np.array(c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO)
            self.viewer_state.camera_handles[idxs[idx]].wxyz = R.wxyz

            if idx > missed_depro:
                if start_idx == 0:
                    points_homog = torch.cat([self.deprojected_queue[idx-missed_depro-1], torch.ones((self.deprojected_queue[idx-missed_depro-1].shape[0], 1), device='cuda:0')], dim=1)
                    self.deprojected_queue[idx-missed_depro-1] = (torch.matmul(BA_deltas[idx].to('cuda:0'), points_homog.T).T)[:, :3]
                else:
                    points_homog = torch.cat([self.deprojected_queue[idx-start_idx], torch.ones((self.deprojected_queue[idx-start_idx].shape[0], 1), device='cuda:0')], dim=1)
                    self.deprojected_queue[idx-start_idx] = (torch.matmul(BA_deltas[idx].to('cuda:0'), points_homog.T).T)[:, :3]
        self.start_idx = len(self.pipeline.datamanager.train_dataset)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            clip_out_queue=self.clip_out_queue,
        )

        self.pipeline.train_lerf = ViewerButton(name="Train LERF", cb_hook=self.handle_train_lerf)
        self.pipeline.start_diff = ViewerButton(name="Start Differencing", cb_hook=self.handle_diff)

        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        self.viewer_log_path = viewer_log_path

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self
            )
        )
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()
        self._init_viewer_state()

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        # profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()

        return Optimizers(optimizer_config, param_groups)

    # @profile
    def train(self) -> None:
        print("IM IN")
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )
        BA_flag = False

        rclpy.init(args=None)
        if self.multicam:
            trainer_node = TricamTrainerNode(self)
            print("multicam trainer node up")
        else:
            trainer_node = TrainerNode(self)

        parser_scale_list = []
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            num_add = 1
            self.imgidx = 0
            
            while True:
                rclpy.spin_once(trainer_node,timeout_sec=0.00)

                has_image_add = len(self.image_add_callback_queue) > 0
                pp_sig = False
                prev_poses = None
                if has_image_add:
                    #Not sure if we want to loop till the queue is empty or not
                    msg, points, colors, prev_poses, pp_sig = self.image_add_callback_queue.pop(0)

                    # if we are actively calculating diff for the current scene,
                    # we don't want to add the image to the dataset unless we are sure.
                    if pp_sig is False:
                        if self.calculate_diff:
                            # TODO: Kishore and Justin
                            # raise NotImplementedError
                            self.query_diff_queue.append(msg)

                        else:
                            self.add_to_clip_queue.append(msg)
                            self.image_process_queue.append((msg, points, colors))
                            self.imgidx += 1

                        if not self.done_scale_calc:
                            parser_scale_list.append(msg.pose)
                        
                while len(self.image_process_queue) > 0:
                    message, pts, clrs = self.image_process_queue.pop(0)
                    self.process_image(message, step, pts, clrs)
                
                if self.train_lerf and len(self.add_to_clip_queue) > 0:
                    self.add_img_callback(self.add_to_clip_queue.pop(0))

                if self.train_lerf and not self.clip_out_queue.empty():
                    print("adding clip pyramid embeddings")
                    self.pipeline.add_to_clip(self.clip_out_queue.get(), step)

                if pp_sig:
                        BA_flag = True
                        new_poses = torch.Tensor(prev_poses).reshape(-1,7)
                        BA_deltas = self.pipeline.datamanager.train_dataset.add_BA_poses(new_poses)
                        self.update_poses(BA_deltas, self.start_idx)

                if self.training_state == "paused":
                    time.sleep(0.01)
                    continue
                
                if not self.done_scale_calc and (len(parser_scale_list)<5):
                    time.sleep(0.01)
                    continue

                #######################################
                # Starting training
                #######################################

                # Create scene scale based on the images collected so far. This is done once.
                if not self.done_scale_calc:
                    # print("Scale calc")
                    self.done_scale_calc = True
                    from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
                    poses = [np.concatenate([ros_pose_to_nerfstudio(p),np.array([[0,0,0,1]])],axis=0) for p in parser_scale_list]
                    poses = torch.from_numpy(np.stack(poses).astype(np.float32))#TODO THIS LINE WRONG
                    poses, transform_matrix = auto_orient_and_center_poses(
                        poses,
                        method='up',
                        center_method='poses'
                    )
                    # scale_factor = 1.0
                    # scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
                    scale_factor = 1.0
                    print(scale_factor)
                    self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform = transform_matrix
                    self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform = transform_matrix
                    self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale = scale_factor
                    self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale = scale_factor
                    # Updated old cameras (pre scale calc) with transform and scale
                    idxs = list(self.viewer_state.camera_handles.keys())
                    for idx in range(len(self.pipeline.datamanager.train_dataset)):
                        C = self.pipeline.datamanager.train_dataset.cameras
                        scale = self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
                        H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
                        c2w = C.camera_to_worlds[idx,...]
                        row = torch.tensor([[0,0,0,1]],dtype=torch.float32,device=c2w.device)
                        c2w= torch.matmul(torch.cat([H,row]),torch.cat([c2w,row]))[:3,:]
                        c2w[:3,3] *= scale  #* VISER_NERFSTUDIO_SCALE_RATIO
                        C.camera_to_worlds[idx,...] = c2w

                        R = vtf.SO3.from_matrix(c2w[:3, :3])
                        R = R @ vtf.SO3.from_x_radians(np.pi)
                        self.viewer_state.camera_handles[idxs[idx]].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
                        self.viewer_state.camera_handles[idxs[idx]].wxyz = R.wxyz
                    print("************\nDone scale calc\n************")
                
                if len(self.pipeline.datamanager.train_dataset) <= 1:
                    time.sleep(0.01)
                    continue

                # Check if we have an image to process, and add *all of them* to the dataset per iteration.

                step +=1 
                

                with self.train_lock:
                    #TODO add the image diff stuff here
                    if step > 5*self.pipeline.model.config.warmup_length:
                        while len(self.query_diff_queue) > self.query_diff_size:
                            self.process_query_diff(self.query_diff_queue.pop(0), step)
                    #######################################
                    # Normal training loop
                    #######################################
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                            
                        # start = time.time()
                        loss, loss_dict, metrics_dict = self.train_iteration(step)
                        
                        # add deprojected gaussians from monocular depth
                        expain = []
                        for group, _ in self.pipeline.model.get_gaussian_param_groups().items():
                            if group == 'lerf':
                                continue
                            expain.append("exp_avg" in self.optimizers.optimizers[group].state[self.optimizers.optimizers[group].param_groups[0]["params"][0]].keys())
                        if all(expain) and BA_flag:
                            self.pipeline.model.deprojected_new.extend(pop_n_elements(self.deprojected_queue, num_add))
                            self.pipeline.model.colors_new.extend(pop_n_elements(self.colors_queue, num_add))

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )
                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

                #After a training loop is done we check to calc metric
                if self.calulate_metrics:
                    self.calc_metric()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            (self.config.is_viewer_enabled() or self.config.is_viewer_beta_enabled())
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state is not None and self.pipeline.datamanager.train_dataset is not None
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state="training",
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")
        CONSOLE.print("Use ctrl+c to quit", justify="center")
        while True:
            time.sleep(0.01)

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=self.world_size * train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    # @profile
    def train_iteration(self, step: int) -> TRAIN_ITERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        # start = time.time()
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            # end = time.time()
            # elapsed = str((end-start)*1e3)
            # print("get_train_loss time: "+ elapsed + "(ms)")
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # print(self.pipeline.print_num_means())
        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)