import json
import os
from pathlib import Path

import numpy as np
import torch
from l3gs.data.utils.feature_dataloader import FeatureDataloader
from l3gs.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from l3gs.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from l3gs.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork


from tqdm import tqdm

import torch.multiprocessing as mp
import queue
import time


class PyramidEmbeddingDataloader(FeatureDataloader, mp.Process):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor = None,
        cache_path: str = None,
        network: BaseImageEncoderConfig = OpenCLIPNetworkConfig(),
        out_queue: mp.Queue = None
    ):
        mp.Process.__init__(self)
        FeatureDataloader.__init__(self, cfg, device, image_list, cache_path)
        self.device = device
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        

        self.network = network
        self.embed_size = self.network.setup().embedding_dim
        self.data_dict = {}
        self.out_queue = out_queue
        self.in_queue = mp.Queue()
        self.cfg = cfg

        print("inside init Pyramid Process")

    def __call__(self, img_points, scale=None):
        if scale is None:
            return self._random_scales(img_points)
        else:
            return self._uniform_scales(img_points, scale)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        raise ValueError("No cache in dynamic loader")
    def save(self):
        pass

    def create(self, img_list, model):
        self.tile_sizes = torch.linspace(*self.cfg["tile_size_range"], self.cfg["tile_size_res"]).to(self.device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), self.cfg["stride_scaler"]) for tr in self.tile_sizes]
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": self.cfg["image_shape"],
                    "model_name": model.name,
                },
                device=self.device,
                model=model,
                image_list=[],
            )
            self.data_dict[i].create(None)

    def add_images(self, image_list):
        for img in image_list:
            self.enqueue_image(img)

    def _random_scales(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        # return: (B, 512), some random scale (between 0, 1)
        img_points = img_points.to(self.device)
        random_scale_bin = torch.randint(self.tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], dtype=torch.float16, device=self.device)

        stepsize = (self.tile_sizes[1] - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)

        for i in range(len(self.tile_sizes) - 1):
            ids = img_points[random_scale_bin == i]
            bottom_interp[random_scale_bin == i] = self.data_dict[i](ids)
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](ids)
        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    def _uniform_scales(self, img_points, scale):
        scale = scale.to(self.device)
        scale_bin = torch.floor(
            (scale - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0]) * (self.tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - self.tile_sizes[scale_bin]) / (
            self.tile_sizes[scale_bin + 1] - self.tile_sizes[scale_bin]
        )
        interp_lst = torch.stack([interp(img_points) for interp in self.data_dict.values()])
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds],
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).half().to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale
    
    def run(self):
        print("Starting RUN", flush=True)
        # model = OpenCLIPNetwork(OpenCLIPNetworkConfig(
        #         clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512))
        print("clip device",self.device)
        model = OpenCLIPNetworkConfig(device=self.device).setup()
        print("Setup Model")
        self.create(None, model)
        print("Create Model")

        running = True
        j = 0
        while running:
            img_batch = []
            while True:
                try:
                    # print(self.in_queue.qsize(), "queue size")
                    img = self.in_queue.get(timeout=.01)  # img is a tuple (image, index)
                    # print("got image")
                    
                except queue.Empty:
                    break
                if img is None:
                    running = False
                    print("PyramidEmbeddingProcess DONE")
                    break
                
                # Append the whole data to img_batch
                img_batch.append(img)

            if len(img_batch) == 0:
                continue
            start = time.time()
            for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
                self.data_dict[i].add_images(img_batch)
            
            assert len(self.data_dict) != 0

            for _ in img_batch:
                updates = []
                for i, tr in enumerate(self.tile_sizes):
                    updates.append(self.data_dict[i].data[j:j+1,...])

                self.out_queue.put(updates)
                j+=1
            
            print(f"PyramidEmbeddingProcess took {time.time()-start} seconds")
            
    def kill(self):
        self.in_queue.put(None)

    def enqueue_image(self, img):
        # print("enqueue image for pyramid...?")
        self.in_queue.put(img)
