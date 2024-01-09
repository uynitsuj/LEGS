import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

# import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
# from torch.nn.parallel import DistributedDataParallel as DDP

# from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.utils.math import intersect_aabb, intersect_obb
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
# from nerfstudio.data.scene_box import OrientedBox

from l3gs.data.L3GS_datamanager import (
    L3GSDataManager,
    L3GSDataManagerConfig,
)

# import viser
# import viser.transforms as vtf
# import trimesh
# import open3d as o3d
# import cv2
from copy import deepcopy

from dataclasses import dataclass, field
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer_beta.viewer_elements import ViewerCheckbox
from nerfstudio.models.base_model import ModelConfig
# from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig
from l3gs.model.ll_gaussian_splatting import LLGaussianSplattingModelConfig
from l3gs.monodepth.zoedepth_network import ZoeDepthNetworkConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torchvision.transforms.functional import resize
from nerfstudio.configs.base_config import InstantiateConfig
# from lerf.utils.camera_utils import deproject_pixel, get_connected_components, calculate_overlap, non_maximum_suppression
from l3gs.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from gsplat.sh import SphericalHarmonics, num_sh_bases

from typing import Literal, Type, Optional, List, Tuple, Dict
# import lerf.utils.query_diff_utils as query_diff_utils
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np 
import math

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )
def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

@dataclass
class L3GSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: L3GSPipeline)
    """target class to instantiate"""
    datamanager: L3GSDataManagerConfig = L3GSDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = LLGaussianSplattingModelConfig()
    """specifies the model config"""
    depthmodel:InstantiateConfig = ZoeDepthNetworkConfig()
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""


class L3GSPipeline(VanillaPipeline):
    def __init__(
        self,
        config: L3GSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        highres_downscale : float = 4.0,
        use_clip : bool = False,
        model_name : str = "dino_vits8",
        dino_thres : float = 0.4, 
        clip_out_queue : Optional[mp.Queue] = None,
        # dino_out_queue : Optional[mp.Queue] = None,
        use_depth = True, 
        use_rgb = False, 
        use_vit = False, 
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.clip_out_queue = clip_out_queue
        # self.dino_out_queue = dino_out_queue
        self.datamanager: L3GSDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            network=self.config.network,
            clip_out_queue=self.clip_out_queue,
            # dino_out_queue=self.dino_out_queue,
        )
        self.datamanager.to(device)
        self.image_encoder: BaseImageEncoder = config.network.setup()
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            image_encoder=self.image_encoder,
            grad_scaler=grad_scaler,
            datamanager=self.datamanager,
        )
        self.model.to(device)

        self.depthmodel = config.depthmodel.setup()

        # self.world_size = world_size
        # if world_size > 1:
        #     self._model = typing.cast(LERFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
        #     dist.barrier(device_ids=[local_rank])

        # self.highres_downscale = highres_downscale
        
        # self.use_rgb = use_rgb
        # self.use_clip = use_clip 
        # self.use_vit = use_vit
        # self.use_depth = use_depth
        # # only one of use rgb, use clip, and use depth can be true 
        # assert (self.use_rgb + self.use_clip + self.use_depth + self.use_vit) == 1, "only one of use_rgb, use_clip, and use_depth can be true"
        # self.model_name = model_name
        # # self.diff_checkbox = ViewerCheckbox("Calculate diff",False)
        
        # if not self.use_clip:
        #     assert model_name in query_diff_utils.model_params.keys(), "model name not found"
        #     self.extractor = ViTExtractor(
        #         model_name, 
        #         query_diff_utils.model_params[model_name]['dino_stride'],
        #     )
        #     self.dino_thres = dino_thres
            
        self.img_count = 0


    # this only calcualtes the features for the given image
    def add_image(
        self,
        img: torch.Tensor, 
        pose: Cameras, 
    ):
        # if self.diff_checkbox.value:
        #     heat_map = self.query_diff(img, pose)
        #     lerf_output = query_diff_utils.get_lerf_outputs(pose.to(self.device), self, 1.0)
        # fig, ax = plt.subplots(3)
        # ax[0].imshow(img.detach().cpu().numpy())
        #     ax[1].imshow(heat_map.detach().cpu().numpy().squeeze())
        #     ax[2].imshow(lerf_output["rgb"].detach().cpu().numpy())
        # plt.show()
        #     boxes = self.heatmaps2box([heat_map], [pose], [lerf_output["depth"]])
        #     print(boxes)
            # self.mask_volume(boxes) #This will deal with the masks in the datamanager
        print("Adding image to DM",pose.camera_to_worlds[:3,3].flatten())
        self.datamanager.add_image(img, pose)
        self.img_count += 1

    # this actually adds the image to the datamanager + dataset...?
    @profile
    def process_image(
        self,
        img: torch.Tensor, 
        pose: Cameras, 
        clip: dict,
        dino,
    ):
        
        self.datamanager.process_image(img, pose, clip, dino)
        # self.datamanager.train_pixel_sampler.nonzero_indices = torch.nonzero(self.datamanager.train_dataset.mask_tensor[0:len(self.datamanager.train_dataset), ..., 0].to(self.device), as_tuple=False)

    # def print_num_means(self):
        # print(self.model.means.shape[0])

    def monodepth_inference(self, image):
        depth = self.depthmodel.get_depth(image)
        # plt.figure()
        # plt.subplot(211)
        # plt.imshow(image)

        # plt.subplot(212)
        # plt.imshow(depth.cpu().detach().numpy()[0,0,:,:], cmap='magma')
        # plt.show()
        # print(depth)
        return depth
    
    # def add_deprojected_means(self, deprojected, colors, optimizers):
    #     self.model.add_deprojected_means(deprojected, colors, optimizers)