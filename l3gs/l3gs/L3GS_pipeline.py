import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.math import intersect_aabb, intersect_obb
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
import trimesh
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
from l3gs.data.utils.sequential_patch_embedding_dataloader import SequentialPatchEmbeddingDataloader
# from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig
from l3gs.model.ll_gaussian_splatting import LLGaussianSplattingModelConfig
from l3gs.monodepth.zoedepth_network import ZoeDepthNetworkConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torchvision.transforms.functional import resize
from nerfstudio.configs.base_config import InstantiateConfig
# from lerf.utils.camera_utils import deproject_pixel, get_connected_components, calculate_overlap, non_maximum_suppression
from l3gs.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from gsplat.sh import SphericalHarmonics, num_sh_bases

from l3gs.data.scene_box import SceneBox, OrientedBox
from typing import Literal, Type, Optional, List, Tuple, Dict
import l3gs.query_diff_utils as query_diff_utils
from l3gs.L3GS_utils import Utils as U
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import math
from scipy.spatial.distance import cdist
import open3d as o3d

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
        use_clip : bool = True,
        model_name : str = "dino_vits8",
        # dino_thres : float = 0.4, 
        clip_out_queue : Optional[mp.Queue] = None,
        # dino_out_queue : Optional[mp.Queue] = None,
        use_depth = True, 
        use_rgb = True, 
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
        self.plot_verbose = False
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


    def get_clip_patchloader(self, image, pipeline, image_scale):
        # clip_cache_path = Path("dummy_cache2.npy")
        # import time
        # model_name = str(time.time())
        # image = image.permute(2,0,1)[None,...]
        # patchloader = SequentialPatchEmbeddingDataloader(
        #     cfg={
        #         "tile_ratio": image_scale,
        #         "stride_ratio": .25,
        #         "image_shape": image.shape[2:4],
        #         "model_name": model_name,
        #     },
        #     device='cuda:0',
        #     model=self.image_encoder,
        #     image_list=[],
        #     cache_path=clip_cache_path,
        # )
        # import pdb; pdb.set_trace()
        # patchloader.create(None)
        # patchloader.add_images(image)
        # return patchloader

        clip_cache_path = Path("dummy_cache2.npy")
        import time
        model_name = str(time.time())
        image = image.permute(2,0,1)[None,...]
        patchloader = SequentialPatchEmbeddingDataloader(
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

    def get_grid_embeds_patch(self, patchloader, rn, cn, im_h, im_w, img_scale):
        "create points, which is a meshgrid of x and y coordinates, with a z coordinate of 1.0"
        r_res = im_h // rn
        c_res = im_w // cn
        points = torch.stack(torch.meshgrid(torch.arange(0, im_h,r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
        points = torch.cat([torch.zeros((*points.shape[:-1],1), dtype=torch.int64, device='cuda'), points],dim=-1)
        embeds = patchloader(points.view(-1,3))
        return embeds, points

    def get_2d_embeds(self, image: torch.Tensor, scale: float, pipeline):
        # pyramid = get_clip_pyramid(image,pipeline=pipeline,image_scale=scale)
        # embeds,points = get_grid_embeds(pyramid,image.shape[0]//resolution,image.shape[1]//resolution,image.shape[0],image.shape[1],scale)
        patchloader = self.get_clip_patchloader(image, pipeline, scale)
        embeds, points = self.get_grid_embeds_patch(patchloader, image.shape[0] * scale,image.shape[1] * scale, image.shape[0], image.shape[1], scale)
        return embeds, points

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

        # Up-resolution
        # depth = cv2.resize(np.array(depth.cpu()), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        depth = F.interpolate(depth, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)

        return depth

    def query_diff_clip(self, image: torch.Tensor, pose: Cameras, image_scale: float = 0.25, thres: float = 0.5, vis_verbose: bool = False):
        clip_ground_truth_image = torch.norm(image, dim=-1).reshape((120, 212))
        clip_ground_truth_image = clip_ground_truth_image - torch.min(clip_ground_truth_image)
        clip_ground_truth_image = clip_ground_truth_image / (torch.max(clip_ground_truth_image) - torch.min(clip_ground_truth_image))
        clip_ground_truth_image = clip_ground_truth_image.to(torch.float32)
        clip_ground_truth_image_resized_cv2 = cv2.resize(clip_ground_truth_image.cpu().detach().numpy(), (848,480), interpolation=cv2.INTER_AREA)
        plt.imsave('clip_ground_truth.png', clip_ground_truth_image_resized_cv2)

        gsplat_outputs = self.model.get_outputs_full_clip(pose.to(self.device))
        gsplat_outputs['rgb'] = gsplat_outputs['rgb'].detach()
        gsplat_outputs['depth'] = gsplat_outputs['depth'].detach()

        if 'clip' not in gsplat_outputs.keys():
            return torch.zeros_like(image), torch.zeros_like(image), gsplat_outputs
                
        gsplat_outputs['clip'] = gsplat_outputs['clip'].detach()
        gsplat_outputs['clip_scale'] = gsplat_outputs['clip_scale'].detach()

        # rendered clip
        clip_output = gsplat_outputs['clip'].detach()

        # rendered image clip
        rendered_image = gsplat_outputs['rgb'].detach()
        rendered_image_clip, rendered_points = self.get_2d_embeds(rendered_image, image_scale.item(), self)
        
        # shift and renorm
        shift = clip_output - rendered_image_clip
        diff = image - shift
        renormed_diff = diff - torch.min(diff)
        renormed_diff = diff / (torch.max(diff) - torch.min(diff))

        heatmap = torch.norm(renormed_diff, dim=-1).reshape((120, 212))
        heatmap_resized_cv2 = cv2.resize(heatmap.cpu().detach().numpy(), (848,480), interpolation=cv2.INTER_AREA)
        plt.imsave('clip_diff_renormed.png', heatmap_resized_cv2)

        # compute final clip diff
        final_clip_diff = image * renormed_diff
        heatmap = torch.norm(final_clip_diff, dim=-1).reshape((120, 212))
        heatmap_mask = heatmap > 0.43 # thres
        heatmap_resized_cv2 = cv2.resize(heatmap.cpu().detach().numpy(), (848,480), interpolation=cv2.INTER_AREA)
        heatmap_mask_resized_cv2 = cv2.resize(heatmap_mask.cpu().detach().numpy().astype(np.uint8), (848,480), interpolation=cv2.INTER_AREA)
        plt.imsave('clip_diff_final.png', heatmap_resized_cv2)
        plt.imsave('clip_diff_final_mask.png', heatmap_mask_resized_cv2)

        # shifted_embeds = image + shift
        # shifted_embeds = shifted_embeds / shifted_embeds.norm(dim=-1, keepdim=True)
        # baselined_diff = -torch.einsum('ijk,ijk->ij', renormed_diff, clip_output)
        
        if vis_verbose:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(-torch.einsum('ijk,ijk->ij', rendered_image_clip, clip_output).detach().cpu().numpy())
            axes[0].set_title("not-baselined diff")
            axes[1].imshow(heatmap.detach().cpu().numpy())
            axes[1].set_title("baselined diff")
            plt.show()

        return heatmap, heatmap_mask, gsplat_outputs
    
    def query_diff_rgb(self, image: torch.Tensor, pose: Cameras, vis_verbose: bool = False, thres: float = 0.5):
        gsplat_outputs = self.model.get_outputs(pose.to(self.device))
        
        rgb_output = gsplat_outputs['rgb']
        plt.imsave('rgb_output.png', rgb_output.detach().cpu().numpy())
        plt.imsave('rgb_ground_truth.png', image.detach().cpu().numpy())
        diff = torch.norm(rgb_output - image, dim=-1)
        plt.imsave('rgb_diff.png', diff.detach().cpu().numpy())
        diff_bool = diff > thres
        kernel = np.ones((3, 3), np.uint8)
        diff_bool_eroded = cv2.erode(diff_bool.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        diff_bool_cleaned = cv2.dilate(diff_bool_eroded, kernel, iterations=1)
        diff_bool_cleaned = torch.from_numpy(diff_bool_cleaned).to(self.device)
        plt.imsave('rgb_diff_bool_cleaned.png', diff_bool_cleaned.cpu().detach().numpy())

        # depth_output = gsplat_outputs['depth'].squeeze(2)
        # plt.imsave('depth_output.png', depth_output.detach().cpu().numpy())
        # diff = torch.norm(depth_output - depth, dim=-1)
        # plt.imsave('depth_diff.png', diff.detach().cpu().numpy())
        # diff_bool = diff > thres
        # plt.imsave('depth_diff_bool.png', diff_bool.detach().cpu().numpy())
        # kernel = np.ones((3, 3), np.uint8)
        # diff_bool_eroded = cv2.erode(diff_bool.cpu().numpy().astype(np.uint8), kernel, iterations=1)
        # diff_bool_cleaned = cv2.dilate(diff_bool_eroded, kernel, iterations=1)
        # diff_bool_cleaned = torch.from_numpy(diff_bool_cleaned).to(self.device)
        # plt.imsave('depth_diff_bool_cleaned.png', diff_bool_cleaned.cpu().detach().numpy())
        
        if vis_verbose:
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(image.detach().cpu().numpy())
            ax[1].imshow(rgb_output.detach().cpu().numpy())
            ax[2].imshow(diff.detach().cpu().numpy())
            # ax[3].imshow(diff_bool.detach().cpu().numpy())
            plt.show()

        return diff, diff_bool_cleaned, gsplat_outputs
    
    def resize_image_to_full_size(self, image: torch.Tensor):
        image_resized = cv2.resize(image.cpu().detach().numpy(), (848,480), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image_resized).to(self.device)
    
    def heatmaps2box(self, 
        heatmaps: List[torch.Tensor], # list of boolean tensors (HxW)
        heatmap_masks: List[torch.Tensor], # list of boolean tensors (HxW)
        images: List[torch.Tensor],
        poses: List[Cameras], 
        depths: List[torch.Tensor], # Nerfstudio depth (distance-depth)
        depth_distance: List[torch.Tensor], # Realsense depth (z-depth) converted to nerfstudio depth (distance-depth)
        gsplat_outputs_list: List[Dict[str, torch.Tensor]],
        nms : float = None, # apply nms
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], trimesh.PointCloud]:
        
        assert len(heatmaps) == len(depths) == len(poses), "length must be equal"
        assert len(heatmaps) == 1, "length must be 1"

        hm, hm_mask, image, pose, depth, gsplat_output = heatmaps[0], heatmap_masks[0], images[0], poses[0], depths[0], gsplat_outputs_list[0]
        if hm.shape[0] == 120:
            hm = self.resize_image_to_full_size(hm)
        if hm_mask.shape[0] == 120:
            hm_mask = self.resize_image_to_full_size(hm_mask.to(torch.float32)).to(torch.bool)
        if image.shape[0] == 120:
            image = self.resize_image_to_full_size(image)
        if depth.shape[0] == 120:
            depth = self.resize_image_to_full_size(depth)
        if gsplat_output['rgb'].shape[0] == 120:
            gsplat_output['rgb'] = self.resize_image_to_full_size(gsplat_output['rgb'])
        if gsplat_output['depth'].shape[0] == 120:
            gsplat_output['depth'] = self.resize_image_to_full_size(gsplat_output['depth'])

        if torch.sum(hm) == 0:
            return [], []
        
        components = U.get_connected_components(hm_mask)
        if len(components) == 0:
            return [], []
        
        largest_component_idx = torch.argmax(torch.tensor([torch.sum(comp) for comp in components]))
        component = components[largest_component_idx].to(self.device)

        plt.imsave('component.png', component.detach().cpu().numpy())
        plt.imsave('depth.png', depth.detach().cpu().numpy())
        plt.imsave('rendered_depth.png', gsplat_output['depth'].squeeze(2).detach().cpu().numpy())

        # depth = depth * self.datamanager.train_dataparser_outputs.dataparser_scale
        depth = gsplat_output['depth'].squeeze(2) / self.datamanager.train_dataparser_outputs.dataparser_scale

        depth = depth * component
        while len(depth.shape) < 4:
            depth = depth.unsqueeze(0)
        included_points, _ = U.deproject_to_RGB_point_cloud(image, depth, pose, 
            self.datamanager.train_dataparser_outputs.dataparser_scale, sampling=False)
        included_points_color = torch.ones_like(included_points)

        # check for outlier rejection
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(included_points.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(included_points_color.detach().cpu().numpy())
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.01)
        intvector = pcd.cluster_dbscan(eps=0.5, min_points=8, print_progress=False)
        intvector = np.asarray(intvector)

        if len(np.unique(intvector)) == 1 and intvector[0] == -1:
            return [], []
        if len(np.unique(intvector)) == 1: # only one cluster, not noise (intvector == 0)
            assert intvector[0] == 0
            points = torch.from_numpy(np.asarray(pcd.points))[intvector == 0]
            points_tr = trimesh.PointCloud(points.numpy())
            points_tr.visual.vertex_colors = np.asarray(pcd.colors)[intvector == 0]
        elif len(np.unique(intvector)) == 2: # only one cluster, and noise (intvector == -1)
            points = torch.from_numpy(np.asarray(pcd.points))[intvector != -1]
            points_tr = trimesh.PointCloud(points.numpy())
            points_tr.visual.vertex_colors = np.asarray(pcd.colors)[intvector != -1]
        else:
            # choose the largest cluster that is not -1
            cluster_sizes = []
            for i in range(np.unique(intvector).shape[0] - 1): # ignore -1
                cluster_sizes.append(np.sum(intvector == i))
            cluster_sizes = np.asarray(cluster_sizes)
            chosen_cluster = np.argmax(cluster_sizes)
            points = torch.from_numpy(np.asarray(pcd.points))[intvector == chosen_cluster]
            points_tr = trimesh.PointCloud(points.numpy())
            points_tr.visual.vertex_colors = np.asarray(pcd.colors)[intvector == chosen_cluster]

        # construct new pcd with only the largest cluster
        second_pcd = o3d.geometry.PointCloud()
        second_pcd.points = o3d.utility.Vector3dVector(included_points.detach().cpu().numpy())
        second_pcd.colors = o3d.utility.Vector3dVector(included_points_color.detach().cpu().numpy())
        second_pcd, ind = second_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.01)
        second_intvector = second_pcd.cluster_dbscan(eps=0.5, min_points=8, print_progress=False)
        second_intvector = np.asarray(second_intvector)

        if len(np.unique(second_intvector)) == 1 and second_intvector[0] == -1:
            return [], []

        list_of_boxes = []
        for i in np.unique(second_intvector): # ignore -1
            if i == -1:
                continue
            points = torch.from_numpy(np.asarray(second_pcd.points))[second_intvector == i]
            points_proj = points.clone()
            
            obox = OrientedBox.from_points(points_proj, device=self.device)

            # OrientedBox.from_points uses PCA to find the orientation of the box, so 
            # the first component is aligned with the z-axis of the scene (because it's the largest variance)
            # obox.T[2] = 0
            # obox.S[0] = 0
            
            # obox_3d = OrientedBox(
            #     R=obox.R,
            #     T=obox.T + torch.Tensor([0, 0, torch.mean(points[:, 2])]).to(self.device),
            #     S=(obox.S + torch.Tensor([torch.max(points[:, 2]) - torch.min(points[:, 2]), 0, 0]).to(self.device)) * 1.5,
            # )
            # obox.S *= self.datamanager.train_dataparser_outputs.dataparser_scale
            # obox.R, obox.T, obox.S = obox.R, obox.T / 10., obox.S / 10.
            list_of_boxes.append(obox)

        points = sum([points_tr], trimesh.PointCloud(vertices=np.array([[0, 0, 0]])))
        return list_of_boxes, points

    def heatmaps2box_old(self, 
        heatmaps: List[torch.Tensor], # list of boolean tensors (HxW)
        heatmap_masks: List[torch.Tensor], # list of boolean tensors (HxW)
        images: List[torch.Tensor],
        poses: List[Cameras], 
        depths: List[torch.Tensor], # Nerfstudio depth (distance-depth)
        depth_distance: List[torch.Tensor], # Realsense depth (z-depth) converted to nerfstudio depth (distance-depth)
        lerf_outputs_list: List[Dict[str, torch.Tensor]],
        nms : float = None, # apply nms
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], trimesh.PointCloud]:
        assert len(heatmaps) == len(depths) == len(poses), "length must be equal"
        
        list_of_boxes = []
        list_of_points_tr = []

        for hm, hm_mask, image, pose, depth, d_distance, l_output in zip(heatmaps, heatmap_masks, images, poses, depths, depth_distance, lerf_outputs_list):
            if torch.sum(hm) == 0:
                continue
            
            components = U.get_connected_components(hm_mask)
            largest_component_idx = torch.argmax(torch.tensor([torch.sum(comp) for comp in components]))
            component = components[largest_component_idx]

            plt.imsave('component.png', component.detach().cpu().numpy())
            plt.imsave('depth.png', depth.detach().cpu().numpy())
            plt.imsave('rendered_depth.png', l_output['depth'].squeeze(2).detach().cpu().numpy())
            component_mask = torch.where(component > 0)

            # depth = depth * self.datamanager.train_dataparser_outputs.dataparser_scale
            depth = l_output['depth'].squeeze(2)

            # included_points = torch.stack([*component_mask, depth[component_mask]], dim=-1)
            depth = depth * component
            while len(depth.shape) < 4:
                depth = depth.unsqueeze(0)
            included_points, _ = U.deproject_to_RGB_point_cloud(image, depth, pose, 
                self.datamanager.train_dataparser_outputs.dataparser_scale, sampling=False)

            # included_points_color = l_output['rgb'][component_mask]
            # included_points_color = image[component_mask]
            included_points_color = torch.ones_like(included_points)

            min_corner = torch.min(included_points, dim=0).values
            max_corner = torch.max(included_points, dim=0).values

            # check for outlier rejection
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(included_points.detach().cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(included_points_color.detach().cpu().numpy())
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.01)
            intvector = pcd.cluster_dbscan(eps=0.5, min_points=8, print_progress=False)
            intvector = np.asarray(intvector)

            if len(np.unique(intvector)) == 1 and intvector[0] == -1:
                continue
            if len(np.unique(intvector)) == 1: # only one cluster, not noise (intvector == 0)
                assert intvector[0] == 0
                points = torch.from_numpy(np.asarray(pcd.points))[intvector == 0]
                points_tr = trimesh.PointCloud(points.numpy())
                points_tr.visual.vertex_colors = np.asarray(pcd.colors)[intvector == 0]
            elif len(np.unique(intvector)) == 2: # only one cluster, and noise (intvector == -1)
                points = torch.from_numpy(np.asarray(pcd.points))[intvector != -1]
                points_tr = trimesh.PointCloud(points.numpy())
                points_tr.visual.vertex_colors = np.asarray(pcd.colors)[intvector != -1]
            else:
                # choose the largest cluster that is not -1
                cluster_sizes = []
                for i in range(np.unique(intvector).shape[0] - 1): # ignore -1
                    cluster_sizes.append(np.sum(intvector == i))
                cluster_sizes = np.asarray(cluster_sizes)
                chosen_cluster = np.argmax(cluster_sizes)
                points = torch.from_numpy(np.asarray(pcd.points))[intvector == chosen_cluster]
                points_tr = trimesh.PointCloud(points.numpy())
                points_tr.visual.vertex_colors = np.asarray(pcd.colors)[intvector == chosen_cluster]

            list_of_points_tr.append(points_tr)

        if len(list_of_points_tr) == 0:
            return [], []
        
        # create bounding box from all the points aggregated using `list_of_points_tr`
        agg_points_tr = sum(list_of_points_tr)
        # cluster the points
        agg_pcd = o3d.geometry.PointCloud()
        agg_pcd.points = o3d.utility.Vector3dVector(agg_points_tr.vertices)
        agg_pcd.colors = o3d.utility.Vector3dVector(agg_points_tr.visual.vertex_colors[:, :3])
        intvector = agg_pcd.cluster_dbscan(eps=0.5, min_points=8, print_progress=False)
        intvector = np.asarray(intvector)

        if len(np.unique(intvector)) == 1 and intvector[0] == -1:
            return [], []
        
        for i in np.unique(intvector): # ignore -1
            if i == -1:
                continue
            points = torch.from_numpy(np.asarray(agg_pcd.points))[intvector == i]

            if len(list_of_boxes) > 0 and len(points) < 30:
                continue

            points_proj = points.clone()
            # points_proj[:, 2] = torch.rand(points_proj.shape[0]) * 0.05
            # points_proj = torch.cat([points_proj, points_proj + torch.Tensor([0, 0, 10.0])], dim=0)

            #### OrientedBox
            obox = OrientedBox.from_points(points_proj, device=self.device)

            # OrientedBox.from_points uses PCA to find the orientation of the box, so 
            # the first component is aligned with the z-axis of the scene (because it's the largest variance)
            # obox.T[2] = 0
            # obox.S[0] = 0
            
            # obox_3d = OrientedBox(
            #     R=obox.R,
            #     T=obox.T + torch.Tensor([0, 0, torch.mean(points[:, 2])]).to(self.device),
            #     S=(obox.S + torch.Tensor([torch.max(points[:, 2]) - torch.min(points[:, 2]), 0, 0]).to(self.device)) * 1.5,
            # )
            # obox.S *= self.datamanager.train_dataparser_outputs.dataparser_scale
            # obox.R, obox.T, obox.S = obox.R, obox.T / 10., obox.S / 10.
            list_of_boxes.append(obox)
            #### end OrientedBox

            #### SceneBox
            # aabb = torch.stack([min_corner, max_corner])
            # sbox = SceneBox(aabb=aabb)
            # list_of_boxes.append(sbox)
            #### end SceneBox

        # # Create a point cloud of the points
        points = sum(list_of_points_tr, trimesh.PointCloud(vertices=np.array([[0, 0, 0]])))
        return list_of_boxes, points
    
    def bbox2gaussians(self, obox: OrientedBox):
        mask_within = obox.within(self.model.means)
        return torch.where(mask_within == True)