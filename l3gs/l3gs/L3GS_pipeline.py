<<<<<<< HEAD
from pathlib import Path
import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional
from l3gs.l3gs.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import Model, ModelConfig
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
import cv2
import math

import L3GS_utils.Utils as U

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

def get_clip_patchloader(image, pipeline, image_scale):
    clip_cache_path = Path("dummy_cache2.npy")
    import time
    model_name = str(time.time())
    image = image.permute(2,0,1)[None,...]
    patchloader = PatchEmbeddingDataloader(
        cfg={
            "tile_ratio": image_scale,
            "stride_ratio": .25,
            "image_shape": list(image.shape[2:4]),
            "model_name": model_name,
        },
        device='cuda:0',
        model=pipeline.image_encoder,
        image_list=image,
        cache_path=clip_cache_path,
    )
    return patchloader

def get_grid_embeds_patch(patchloader, rn, cn, im_h, im_w, img_scale):
    "create points, which is a meshgrid of x and y coordinates, with a z coordinate of 1.0"
    r_res = im_h // rn
    c_res = im_w // cn
    points = torch.stack(torch.meshgrid(torch.arange(0, im_h,r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
    points = torch.cat([torch.zeros((*points.shape[:-1],1),dtype=torch.int64,device='cuda'),points],dim=-1)
    embeds = patchloader(points.view(-1,3))
    return embeds, points

def get_2d_embeds(image: torch.Tensor, scale: float, pipeline):
    # pyramid = get_clip_pyramid(image,pipeline=pipeline,image_scale=scale)
    # embeds,points = get_grid_embeds(pyramid,image.shape[0]//resolution,image.shape[1]//resolution,image.shape[0],image.shape[1],scale)
    patchloader = get_clip_patchloader(image, pipeline=pipeline, image_scale=scale)
    embeds, points = get_grid_embeds_patch(patchloader, image.shape[0] * scale,image.shape[1] * scale, image.shape[0], image.shape[1], scale)
    return embeds, points

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
        # dino_thres : float = 0.4, 
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

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        # self.highres_downscale = highres_downscale
        
        self.use_rgb = use_rgb
        self.use_clip = use_clip 
        self.use_vit = use_vit
        self.use_depth = use_depth
        # # only one of use rgb, use clip, and use depth can be true 
        assert (self.use_rgb + self.use_clip + self.use_depth + self.use_vit) == 1, "only one of use_rgb, use_clip, and use_depth can be true"
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
        pose: Cameras = None, 
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
        # self.datamanager.add_image(img, pose)
        self.datamanager.add_image(img)
        self.img_count += 1

    # this actually adds the image to the datamanager + dataset...?
    # @profile
    def process_image(
        self,
        img: torch.Tensor, 
        pose: Cameras, 
        clip: dict,
        dino,
    ):
        print("Adding image to train dataset",pose.camera_to_worlds[:3,3].flatten())
        
        self.datamanager.process_image(img, pose, clip, dino)
        # self.datamanager.train_pixel_sampler.nonzero_indices = torch.nonzero(self.datamanager.train_dataset.mask_tensor[0:len(self.datamanager.train_dataset), ..., 0].to(self.device), as_tuple=False)

    # def print_num_means(self):
        # print(self.model.means.shape[0])

    def add_to_clip(self, clip: dict, step: int):
        self.datamanager.add_to_clip(clip, step)

    def monodepth_inference(self, image):
        # print(type(image))
        # Down-sample
        down_height = image.shape[0] // 2
        down_width = image.shape[1] // 2
        imagedown = cv2.resize(np.array(image), (down_width, down_height), interpolation=cv2.INTER_AREA)
        
        depth = self.depthmodel.get_depth(imagedown)

        # import pdb; pdb.set_trace()
        # Up-resolution
        # depth = cv2.resize(np.array(depth.cpu()), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth = F.interpolate(depth, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)



        return depth
    

    def query_diff(self, image: torch.Tensor, pose: Cameras, depth, vis_verbose: bool = False):
        if self.use_clip:
            heat_map, gsplat_outputs = self.query_diff_clip(image, pose)
        elif self.use_rgb:
            heat_map, gsplat_outputs = self.query_diff_rgb(image, pose)
        return heat_map, gsplat_outputs
    
    def query_diff_clip(self, image: torch.Tensor, pose: Cameras, image_scale: float = 0.25, vis_verbose: bool = False):
        gsplat_outputs = self.model.get_outputs(pose)
        gsplat_clip_output = gsplat_outputs['clip']

        rendered_image = gsplat_outputs['rgb']
        rendered_embeds, rendered_points = get_2d_embeds(rendered_image, image_scale, self)

        im_h, im_w, _ = image.shape
        r_res, c_res = im_h * image_scale, im_w * image_scale
        image_coords = torch.stack(torch.meshgrid(torch.arange(0, im_h, r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
        image_embeds, image_points = get_2d_embeds(image, image_scale, self)

        shift = gsplat_clip_output - rendered_embeds
        shifted_embeds = image_embeds + shift
        shifted_embeds = shifted_embeds / shifted_embeds.norm(dim=-1, keepdim=True)
        baselined_diff = -torch.einsum('ijk,ijk->ij', shifted_embeds, gsplat_clip_output)

        if vis_verbose:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(-torch.einsum('ijk,ijk->ij', rendered_embeds, gsplat_clip_output).detach().cpu().numpy())
            axes[0].set_title("not-baselined diff")
            axes[1].imshow(baselined_diff.detach().cpu().numpy())
            axes[1].set_title("baselined diff")
            plt.show()

        return baselined_diff, gsplat_outputs



    def query_diff_rgb(self, image: torch.Tensor, pose: Cameras, vis_verbose: bool = False, thres: float = 0.03):
        gsplat_outputs = self.model.get_outputs(pose)
        rgb_output = gsplat_outputs['rgb']

        diff = torch.norm(rgb_output - image, dim=-1)
        diff_bool = diff > thres

        if vis_verbose:
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(image.detach().cpu().numpy())
            ax[1].imshow(rgb_output.detach().cpu().numpy())
            ax[2].imshow(diff.detach().cpu().numpy())
            ax[3].imshow(diff_bool.detach().cpu().numpy())
            plt.show()

        return diff_bool, gsplat_outputs
    
    def heatmaps2gaussians(self, heatmap_masks, gsplat_outputs, poses, depths, images):
        # look at depth where heatmap is activated, find corresponding gaussians?
        #   then how to mask out volumes? do we replace all affected gaussians?
        
        assert len(heatmap_masks) == len(depths) == len(poses), "length must be equal"
        distance_thresh = 0.03
        affected_gaussians = []

        for hm, go, p, d, im in enumerate(heatmap_masks, gsplat_outputs, poses, depths, images):
            masked_depth = d & hm
            deprojected, _ = U.deproject_to_RGB_point_cloud(im, masked_depth, p, self.datamanager.train_dataparser_outputs.dataparser_scale)
            to_flag = torch.where(torch.abs(self.means - deprojected) < distance_thresh)
            flagged = self.means[to_flag] # flags all gaussian means that are close to deprojected points
            affected_gaussians.append(flagged)
        
        return affected_gaussians
=======
import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

# from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import Model, ModelConfig
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
import cv2
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
        # dino_thres : float = 0.4, 
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

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

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
        pose: Cameras = None, 
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
        # self.datamanager.add_image(img, pose)
        self.datamanager.add_image(img)
        # self.img_count += 1

    # this actually adds the image to the datamanager + dataset...?
    # @profile
    def process_image(
        self,
        img: torch.Tensor, 
        pose: Cameras, 
        clip: dict,
        dino,
    ):
        print("Adding image to train dataset",pose.camera_to_worlds[:3,3].flatten())
        
        self.datamanager.process_image(img, pose, clip, dino)
        self.img_count += 1
        # self.datamanager.train_pixel_sampler.nonzero_indices = torch.nonzero(self.datamanager.train_dataset.mask_tensor[0:len(self.datamanager.train_dataset), ..., 0].to(self.device), as_tuple=False)

    # def print_num_means(self):
        # print(self.model.means.shape[0])

    def add_to_clip(self, clip: dict, step: int):
        self.datamanager.add_to_clip(clip, step)

    def monodepth_inference(self, image):
        # print(type(image))
        # Down-sample
        down_height = image.shape[0] // 2
        down_width = image.shape[1] // 2
        imagedown = cv2.resize(np.array(image), (down_width, down_height), interpolation=cv2.INTER_AREA)
        
        depth = self.depthmodel.get_depth(imagedown)

        # import pdb; pdb.set_trace()
        # Up-resolution
        # depth = cv2.resize(np.array(depth.cpu()), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth = F.interpolate(depth, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)



        return depth
>>>>>>> origin/main
