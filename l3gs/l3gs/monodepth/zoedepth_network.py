from dataclasses import dataclass, field
from typing import Tuple, Type
from guppy import hpy
import torch
from torchvision import transforms, datasets
from nerfstudio.configs.base_config import InstantiateConfig
from PIL import Image
import numpy as np
# from nerfstudio.viewer_beta.viewer_elements import ViewerText


@dataclass
class ZoeDepthNetworkConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: ZoeDepthNetwork)
    depth_model: str = 'ZoeD_N' #ZoeD_NK, ZoeD_N, ZoeD_K
    repo = "isl-org/ZoeDepth"
    device: str = 'cuda:0'


class ZoeDepthNetwork():
    def __init__(self, config: ZoeDepthNetworkConfig):
        # super().__init__()
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        self.config = config
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load(self.config.repo, self.config.depth_model, pretrained=True)
        self.model = self.model.to(self.config.device)


    @property
    def name(self) -> str:
        return "depth_{}".format(self.config.depth_model)

    # @profile
    def get_depth(self, image):
        if type(image) == Image or type(image) == np.ndarray:
            image = transforms.ToTensor()(image).to(self.config.device).unsqueeze(0)
        elif type(image) == torch.Tensor:
            image = (image).to(self.config.device).unsqueeze(0)
        else:
            raise Exception("Image type not supported")
        depth = self.model.infer(image)
        depth.detach().cpu()

        # h = hpy()
        # print(h.heap())
        return depth