from l3gs.data.utils.feature_dataloader import FeatureDataloader
from l3gs.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from l3gs.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from l3gs.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from l3gs.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
import torch.multiprocessing as mp
import torch
import nerfstudio.utils.poses as pose_utils
import numpy as np
import scipy.spatial.transform as transform
import rclpy
from rclpy.node import Node
from lifelong_msgs.msg import ImagePose
from l3gs.L3GS_pipeline import L3GSPipeline
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge  # Needed for converting between ROS Image messages and OpenCV images


class TrainerNode(Node):
    def __init__(self,trainer):
        super().__init__('trainer_node')
        self.trainer_ = trainer
        self.subscription_ = self.create_subscription(ImagePose,"/camera/color/imagepose",self.add_img_callback,100)

        # self.subscription_ = self.create_subscription(ImagePose,"/sim_realsense",self.add_img_callback,100)

    def add_img_callback(self,msg):
        print("Appending imagepose to queue",flush=True)
        self.trainer_.image_add_callback_queue.append(msg)


def main():
    """Whether to cache images in memory. If "numpy", caches as numpy arrays, if "torch", caches as torch tensors."""
    patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)
    """The range of tile sizes to sample from for patch-based training"""
    patch_tile_size_res: int = 7
    """The number of tile sizes to sample from for patch-based training"""
    patch_stride_scaler: float = 0.5
    """The stride scaler for patch-based training"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""
    clip_downscale_factor: int = 2
    """The downscale factor for the clip pyramid"""
    h = 480
    w = 848
    clip_out_queue = mp.Queue()
    clip_interpolator = PyramidEmbeddingDataloader(
            image_list=[],
            device='cuda:0',
            cfg={
                "tile_size_range": list(patch_tile_size_range),
                "tile_size_res": patch_tile_size_res,
                "stride_scaler": patch_stride_scaler,
                "image_shape": [h,w],
            },
            cache_path=None,
            out_queue= clip_out_queue,
            # network=network,
        )
    clip_interpolator.start()
    clip_interpolator.device = 'cuda:0' #??
    clip_interpolator.create(None, network.setup())

    rclpy.init(args=None)
    trainer_node = TrainerNode()
    parser_scale_list = []
    # with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
        # num_iterations = self.config.max_num_iterations
    step = 0
    num_add = 4
    imgidx = 0
    
    while True:
        rclpy.spin_once(trainer_node,timeout_sec=0.00)

        has_image_add = len(trainer_node.image_add_callback_queue) > 0
        if has_image_add:
            #Not sure if we want to loop till the queue is empty or not
            msg = trainer_node.image_add_callback_queue.pop(0)

            cvbridge = CvBridge()
            image_data = torch.tensor(cvbridge.imgmsg_to_cv2(msg.img,'rgb8'),dtype = torch.float32)/255.
            

            # image_process_queue.append(msg)
            imgidx += 1

