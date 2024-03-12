from typing import Union

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.process_data.colmap_utils import qvec2rotmat


class L3GSDataset(InputDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    In reality we could just store everything directly in ROSDataloader, but this
    would require rewritting more code than its worth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones).
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert ("num_images" in dataparser_outputs.metadata.keys())
        self.num_images = self.metadata["num_images"]
        self.image_height = self.metadata['image_height']
        self.image_width = self.metadata['image_width']
        self.depth_height = self.metadata['depth_height']
        self.depth_width = self.metadata['depth_width']
        assert self.num_images > 0
        self.device = device

        self.cameras = self.cameras.to(device=self.device)

        self.image_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
        ).to(self.device)

        self.depth_tensor = torch.ones(
            self.num_images, self.depth_height, self.depth_width, 1, dtype=torch.float32
        ).to(self.device)

        self.image_indices = torch.arange(self.num_images)

        self.stage = [0]

        self.mask_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 1, dtype=torch.uint8
        ).to(self.device)

        self.cur_size = 0

        self.BA_poses = None

    def __len__(self):
        return self.cur_size

    def add_image(self,img,depth,cam):
        if self.cur_size == 0:
            self.image_height = cam.height
            self.image_width = cam.width
            self.depth_height = cam.height
            self.depth_width = cam.width
            self.image_tensor = torch.ones(
                self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
            ).to(self.device)
            self.depth_tensor = torch.ones(
                self.num_images, self.depth_height, self.depth_width, 1, dtype=torch.float32
            ).to(self.device)

        assert self.cur_size +1 < self.num_images, "Overflowed number of imgs in dataset"
        #set the pose of the camera
        c2w = cam.camera_to_worlds
        H = self._dataparser_outputs.dataparser_transform
        row = torch.tensor([[0,0,0,1]],dtype=torch.float32,device=c2w.device)
        c2w= torch.matmul(torch.cat([H,row]),torch.cat([c2w,row]))[:3,:]
        c2w[:3,3] *= self._dataparser_outputs.dataparser_scale
        self.cameras.camera_to_worlds[self.cur_size,...] = c2w # cam.camera_to_worlds
        self.cameras.fx[self.cur_size] = cam.fx
        self.cameras.cx[self.cur_size] = cam.cx
        self.cameras.fy[self.cur_size] = cam.fy
        self.cameras.cy[self.cur_size] = cam.cy
        self.cameras.distortion_params[self.cur_size] = cam.distortion_params
        self.cameras.height[self.cur_size] = cam.height
        self.cameras.width[self.cur_size] = cam.width
        self.image_tensor[self.cur_size,...] = img
        self.depth_tensor[self.cur_size,...] = depth.unsqueeze(-1)
        self.cur_size += 1
        # import pdb; pdb.set_trace()
        # from torch.nn import functional as F
        # torch_img = img.permute(2, 0, 1).unsqueeze(0).float()
        # img = F.interpolate(torch_img,(self.image_height, self.image_width),mode='bilinear').squeeze(0).squeeze(0)
        # img = img.permute(1, 2, 0)
        # self.image_tensor[self.cur_size,...] = img
        # torch_depth = depth.unsqueeze(0).unsqueeze(0).float()
        # depth = F.interpolate(torch_depth,(self.depth_height,self.depth_width),mode='bilinear').squeeze(0).squeeze(0)
        # self.depth_tensor[self.cur_size,...] = depth.unsqueeze(-1)
        # self.cur_size += 1

    def add_BA_poses(self, poses):
        if self.BA_poses is None:
            print("First Bundle Adjustment Poses Added")
        self.BA_poses = poses
        BA_deltas = []
        # import pdb; pdb.set_trace()
        for idx in range(self.cur_size):

            new_posi = torch.tensor([self.BA_poses[idx, 0], self.BA_poses[idx, 1], self.BA_poses[idx, 2]])
            new_quat = torch.tensor([self.BA_poses[idx, 6], self.BA_poses[idx, 3], self.BA_poses[idx, 4], self.BA_poses[idx, 5]])
            new_R = torch.tensor(qvec2rotmat(new_quat))
            
            formatted_R = torch.zeros((3, 3), dtype=torch.float32, device=new_R.device)
            formatted_R[:, 0] = new_R[:, 0]
            formatted_R[:, 1] = -new_R[:, 1]
            formatted_R[:, 2] = -new_R[:, 2]

            new = torch.cat([formatted_R, new_posi.unsqueeze(0).transpose(1, 0)], dim=1).to(self.device)

            H = self._dataparser_outputs.dataparser_transform
            row = torch.tensor([[0,0,0,1]],dtype=torch.float32,device=new.device)
            c2w= torch.matmul(torch.cat([H,row]),torch.cat([new,row]))[:3,:]
            c2w[:3,3] *= self._dataparser_outputs.dataparser_scale

            # find transform from self.cameras.camera_to_worlds[idx, ...] to c2w
            old_c2w = self.cameras.camera_to_worlds[idx, ...]
            old_c2w = torch.tensor(old_c2w, dtype=torch.float32, device=self.device)
            delta_xyz = c2w[:3, 3] - old_c2w[:3, 3]
            delta_so3 = torch.matmul(old_c2w[:3, :3].transpose(1, 0), c2w[:3, :3])
            delta_se3 = torch.eye(4, dtype=torch.float32, device=self.device)
            delta_se3[:3, :3] = delta_so3
            delta_se3[:3, 3] = delta_xyz
            BA_deltas.append(delta_se3)
            self.cameras.camera_to_worlds[idx, ...] = c2w
        return BA_deltas
    
    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = {"image_idx": idx, "image": self.image_tensor[idx], "depth": self.depth_tensor[idx], "mask": self.mask_tensor[idx]}
        return data