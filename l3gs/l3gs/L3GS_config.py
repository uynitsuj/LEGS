"""
L3GS configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from l3gs.encoders.openclip_encoder import OpenCLIPNetworkConfig
# from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig
from l3gs.model.ll_gaussian_splatting import LLGaussianSplattingModelConfig
from l3gs.monodepth.zoedepth_network import ZoeDepthNetworkConfig
from l3gs.monodepth.midas_network import MidasDepthNetworkConfig

from l3gs.L3GS_trainer import TrainerConfig
from l3gs.L3GS_pipeline import L3GSPipelineConfig
from l3gs.data.L3GS_datamanager import L3GSDataManagerConfig, L3GSDataManager
from l3gs.data.L3GS_dataparser import L3GSDataParserConfig
from l3gs.data.L3GS_dataset import L3GSDataset


l3gs_method = MethodSpecification(
    config = TrainerConfig(
        method_name="l3gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=100000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps = {'camera_opt': 100,'color':10,'shs':10},
        pipeline=L3GSPipelineConfig(
            datamanager=L3GSDataManagerConfig(
                _target=L3GSDataManager[L3GSDataset],
                dataparser=L3GSDataParserConfig(),
                
                #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
                # image_encoder=CLIPNetworkConfig(
                #     clip_model_type="ViT-B/16", clip_n_dims=512
                # )
            ),
            model=LLGaussianSplattingModelConfig(),
            network=OpenCLIPNetworkConfig(
                    clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512, device='cuda:0'
            ),
            depthmodel=ZoeDepthNetworkConfig(device='cuda:0'),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "color": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=30000,
                ),
            },
            # "shs": {
            #     "optimizer": AdamOptimizerConfig(lr=2.5e-3 / 20, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(
            #         lr_final=1e-3 / 20,
            #         max_steps=30000,
            #     ),
            # },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000)
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
             "lerf": {
            "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
        },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer_beta",
    ),
    description="Base config for Lifelong Language Embedded Gaussian Splatting",
)
