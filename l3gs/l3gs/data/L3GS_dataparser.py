"""Data parser for loading ROS parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class L3GSDataParserConfig(DataParserConfig):
    """ROS config file parser config."""

    _target: Type = field(default_factory=lambda: L3GSDataParser)
    """ Path to configuration JSON. """
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    aabb_scale: float = 1.0
    """ SceneBox aabb scale."""
    num_images: int = 1000
    img_height: int = 480
    img_width: int = 848


@dataclass
class L3GSDataParser(DataParser):
    """ROS DataParser"""

    config: L3GSDataParserConfig

    def __init__(self, config: L3GSDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.aabb = config.aabb_scale

    def get_dataparser_outputs(self, split="train"):
        dataparser_outputs = self._generate_dataparser_outputs(split)
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train"):
        """
        This function generates a DataParserOutputs object. Typically in Nerfstudio
        this is used to populate the training and evaluation datasets, but since with
        NSROS Bridge our aim is to stream the data then we only have to worry about
        loading the proper camera parameters and ROS topic names.

        Args:
            split: Determines the data split (not used, but left in place for consistency
                with Nerfstudio)

            num_images: The size limit of the training image dataset. This is used to
                pre-allocate tensors for the Cameras object that tracks camera pose.
        """
        meta = {}

        image_height = self.config.img_height
        image_width = self.config.img_width
        #placeholders
        fx = 1.0
        fy = 1.0
        cx = 1.0
        cy = 1.0

        k1 = meta["k1"] if "k1" in meta else 0.0
        k2 = meta["k2"] if "k2" in meta else 0.0
        k3 = meta["k3"] if "k3" in meta else 0.0
        k4 = meta["k4"] if "k4" in meta else 0.0
        p1 = meta["p1"] if "p1" in meta else 0.0
        p2 = meta["p2"] if "p2" in meta else 0.0
        distort = torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)

        camera_to_world = torch.stack(self.config.num_images * [torch.eye(4, dtype=torch.float32)])[
            :, :-1, :
        ]

        # in x,y,z order
        scene_size = self.aabb
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-scene_size, -scene_size, -scene_size],
                    [scene_size, scene_size, scene_size],
                ],
                dtype=torch.float32,
            )
        )

        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=image_height,
            width=image_width,
            distortion_params=distort,
            camera_type=CameraType.PERSPECTIVE,
        )

        image_filenames = []
        metadata = {
            "num_images": self.config.num_images,
            "image_height": image_height,
            "image_width": image_width,
        }

        # Pointcloud or random

        # metadata.update(self._load_3D_points())

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
    
def _load_3D_points(self):
    """
    Loads the 3D points from depth image or RGBD.
    """
    if self.depth_only:
        raise NotImplementedError

    elif self.RGBD:
        raise NotImplementedError

    else:
        raise NotImplementedError

    return out