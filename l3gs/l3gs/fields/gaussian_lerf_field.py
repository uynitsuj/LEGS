# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Literal, Optional, Tuple
import numpy as np

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from l3gs.field_components.gaussian_lerf_fieldheadnames import GaussianLERFFieldHeadNames
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

try:
    import tinycudann as tcnn
except ImportError:
    pass


class GaussianLERFField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        grid_layers: Tuple[int] = (12, 12),
        grid_sizes: Tuple[Tuple[int]] = (19, 19),
        grid_resolutions: Tuple[int] = ((16, 128), (128, 512)),
        n_features_level: int = 4,
        clip_n_dims: int = 512,
        feature_dims: int = 64,
    ) -> None:
        super().__init__()

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        self.clip_encs = torch.nn.ModuleList(
            [
                GaussianLERFField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i], features_per_level=n_features_level,
                ) for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])
        print("Total output dims: ", tot_out_dims)

        # self.mlp_base_grid = HashEncoding(
        #     num_levels=num_levels,
        #     min_res=base_res,
        #     max_res=max_res,
        #     log2_hashmap_size=log2_hashmap_size,
        #     features_per_level=features_per_level,
        #     implementation=implementation,
        # )

        self.clip_net = tcnn.Network(
            n_input_dims=tot_out_dims+1,
            n_output_dims=clip_n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 3,
            },
        )

        # self.dino_net = tcnn.Network(
        #     n_input_dims=tot_out_dims,
        #     n_output_dims=384,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 256,
        #         "n_hidden_layers": 1,
        #     },
        # )

        # self.clip_feature_net = tcnn.Network(
        #     n_input_dims=feature_dims+1,
        #     n_output_dims=clip_n_dims,
        #     network_config={
        #         "otype": "CutlassMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 264,
        #         "n_hidden_layers": 3,
        #     },
        # )
        #the same above network but with a torch.nn.Sequential of MLP with Relu actiavions between
        # self.clip_feature_net = nn.Sequential(
        #     nn.Linear(feature_dims+1, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, clip_n_dims),
        # )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19, features_per_level=8):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs(self, positions, clip_scales) -> Dict[GaussianLERFFieldHeadNames, Tensor]:
        # random scales, one scale
        outputs = {}
        xs = [e(positions.view(-1, 3)) for e in self.clip_encs]
        x = torch.concat(xs, dim=-1)

        outputs[GaussianLERFFieldHeadNames.HASHGRID] = x.view(positions.shape[0], -1)
        
        clip_pass = self.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1)).view(positions.shape[0], -1)
       
        # encoding = self.mlp_base_grid(positions.view(-1, 3))
        # clip_pass = self.clip_net(torch.cat([encoding, clip_scales.view(-1, 1)], dim=-1))
        outputs[GaussianLERFFieldHeadNames.CLIP] = (clip_pass / clip_pass.norm(dim=-1, keepdim=True)).to(torch.float32)

        # dino_pass = self.dino_net(x).view(positions.shape[0], -1)
        # outputs[GaussianLERFFieldHeadNames.DINO] = dino_pass

        return outputs

    def get_hash(self, positions) -> Tensor:
        encodings = [e(positions.view(-1, 3)) for e in self.clip_encs]
        encoding = torch.concat(encodings, dim=-1)
        return encoding.to(torch.float32)
    
    def get_outputs_from_feature(self, clip_features, clip_scale) -> Dict[GaussianLERFFieldHeadNames, Tensor]:
        outputs = {}
        
        #clip_features is Nx32, and clip scale is a number, I want to cat clip scale to the end of clip_features where clip scale is an int
        # clip_pass = self.clip_feature_net(torch.cat([clip_features, clip_scale.view(-1, 1)], dim=-1))
        clip_pass = self.clip_net(torch.cat([clip_features, clip_scale.view(-1, 1)], dim=-1))

        # print("Max scale: ", clip_scale.max(), "Mean scale: ", clip_scale.mean(), "Min scale: ", clip_scale.min())
        # clip_pass = self.clip_feature_net(clip_features)
        outputs[GaussianLERFFieldHeadNames.CLIP] = (clip_pass / clip_pass.norm(dim=-1, keepdim=True)).to(torch.float32)

        # dino_pass = self.dino_net(clip_features).view(clip_features.shape[0], -1)
        # outputs[GaussianLERFFieldHeadNames.DINO] = dino_pass
        return outputs