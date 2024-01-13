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
class MidasDepthNetworkConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: MidasDepthNetwork)
    depth_model: str = 'DPT_Hybrid' #DPT_Large, DPT_Hybrid, ZoeD_K
    repo = "intel-isl/MiDaS"
    device: str = 'cuda:0'


class MidasDepthNetwork():
    def __init__(self, config: MidasDepthNetworkConfig):
        # super().__init__()
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        self.config = config
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load(self.config.repo, self.config.depth_model, pretrained=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        # if self.config.depth_model == "DPT_Large" or self.config.depth_model == "DPT_Hybrid":
        self.transform = midas_transforms.dpt_transform
        # else:
        #     self.transform = midas_transforms.small_transform
        self.model.to(self.config.device)


    @property
    def name(self) -> str:
        return "depth_{}".format(self.config.depth_model)

    # @profile
    def get_depth(self, image):
        # if type(image) == Image or type(image) == np.ndarray:
        #     image = transforms.ToTensor()(image).to(self.config.device).unsqueeze(0)
        # elif type(image) == torch.Tensor:
            # image = (image).to(self.config.device).unsqueeze(0)
        # else:
        #     raise Exception("Image type not supported")
        # depth = self.model.to(self.config.device).infer(image)
        with torch.no_grad():
            prediction = self.model(self.transform(np.array(image)).to(self.config.device))

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # h = hpy()
        # print(h.heap())
        return prediction