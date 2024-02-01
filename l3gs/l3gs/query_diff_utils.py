# from PIL import Image
# from l3gs.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
# from l3gs.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader

# from nerfstudio.utils.eval_utils import eval_setup
# from pathlib import Path
# from torchvision.transforms.functional import resize
# from tqdm import tqdm
# from typing import Optional
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torchvision.transforms as transforms 

# # TODO: these value needs to be tuned for the new setup
# model_params = {
#     "dino_vits8" : {
#         "dino_stride": 8, 
#         "dino_load_size": (500, 892),
#         "dino_layer": 11, 
#         "dino_facet": "key", 
#         "dino_bin": False,  
#     }, 
#     "dino_vitb8" : {
#         "dino_stride": 8, 
#         "dino_load_size": (500, 892),
#         "dino_layer": 11, 
#         "dino_facet": "key", 
#         "dino_bin": False,  
#     },
#     "dinov2_vits14" : {
#         "dino_stride" : 14, 
#         "dino_load_size" : (504, 896), 
#         "dino_layer" : 11, 
#         "dino_facet" : "key", 
#         "dino_bin" : False, 
#     },
#     "dinov2_vitl14" : {
#         "dino_stride" : 14, 
#         "dino_load_size" : (504, 896), 
#         "dino_layer" : 11, 
#         "dino_facet" : "key", 
#         "dino_bin" : False, 
#     }, 
#     "dinov2_vitg14" : {
#         "dino_stride" : 14, 
#         "dino_load_size" : (504, 896), 
#         "dino_layer" : 39, 
#         "dino_facet" : "key", 
#         "dino_bin" : False, 
#     }, 
#     "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k" : {
#         "dino_stride" : 16, 
#         "dino_load_size" :(496, 880),
#         "dino_layer" : 11, 
#         "dino_facet" : "key", 
#         "dino_bin" : False, 
#     }, 
#     "vit_base_patch16_224.mae" : {
#         "dino_stride" : 16, 
#         "dino_load_size" :(496, 880),
#         "dino_layer" : 11, 
#         "dino_facet" : "key", 
#         "dino_bin" : False, 
#     }
# }

# def get_image_and_camera(pipeline,idx):
#     """
#     This function gets the raw image data and camera parameters for a given index in the train dataset
#     returns image in form (H,W,C) 
#     """
#     assert idx>=0 and idx<len(pipeline.datamanager.train_dataset)
#     datamanager = pipeline.datamanager
#     cameras = datamanager.train_dataparser_outputs.cameras
#     assert idx<cameras.shape[0]
#     c = cameras[idx,...]
#     image = datamanager.train_dataset[idx]['image']
#     return image,c

# def get_clip_pyramid(image, pipeline,image_scale=None):
#     """
#     This function returns the clip pyramid for a given index.
#     """
#     clip_cache_path = Path("dummy_cache.npy")
#     import time
#     model_name = str(time.time())
#     image = image.permute(2,0,1)[None,...]
#     if image_scale is not None:
#         pyramid = PyramidEmbeddingDataloader(
#                 image_list=image,
#                 device="cuda:0",
#                 cfg={
#                     "tile_size_range": [image_scale-.01, image_scale+.01],
#                     "tile_size_res": 2,
#                     "stride_scaler": 0.2,
#                     "image_shape": list(image.shape[2:4]),
#                     "model_name": model_name,
#                 },
#                 cache_path=clip_cache_path,
#                 model = pipeline.image_encoder,
#             )
#     else:
#         pyramid = PyramidEmbeddingDataloader(
#                     image_list=image,
#                     device="cuda:0",
#                     cfg={
#                         "tile_size_range": [0.2, 0.5],
#                         "tile_size_res": 5,
#                         "stride_scaler": 0.20,
#                         "image_shape": list(image.shape[2:4]),
#                         "model_name": model_name,
#                     },
#                     cache_path=clip_cache_path,
#                     model = pipeline.image_encoder,
#                 )
#     return pyramid

# def get_clip_patchloader(image, pipeline,image_scale):
#     clip_cache_path = Path("dummy_cache2.npy")
#     import time
#     model_name = str(time.time())
#     image = image.permute(2,0,1)[None,...]
#     patchloader = PatchEmbeddingDataloader(
#         cfg={
#             "tile_ratio": image_scale,
#             "stride_ratio": .25,
#             "image_shape": list(image.shape[2:4]),
#             "model_name": model_name,
#         },
#         device='cuda:0',
#         model=pipeline.image_encoder,
#         image_list=image,
#         cache_path=clip_cache_path,
#     )
#     return patchloader

# def get_grid_embeds(pyramid,rn,cn,im_h,im_w,img_scale):
#     "create points, which is a meshgrid of x and y coordinates, with a z coordinate of 1.0"
#     r_res = im_h // rn
#     c_res = im_w // cn
#     points = torch.stack(torch.meshgrid(torch.arange(0, im_h,r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
#     points = torch.cat([torch.zeros((*points.shape[:-1],1),dtype=torch.int64,device='cuda'),points],dim=-1)
#     embeds,scale = pyramid(points.view(-1,3),scale=img_scale)
#     embeds = embeds.to("cuda")
#     return embeds,points

# def get_grid_embeds_patch(patchloader,rn,cn,im_h,im_w,img_scale):
#     "create points, which is a meshgrid of x and y coordinates, with a z coordinate of 1.0"
#     r_res = im_h // rn
#     c_res = im_w // cn
#     points = torch.stack(torch.meshgrid(torch.arange(0, im_h,r_res), torch.arange(0,im_w,c_res)), dim=-1).cuda().long()
#     points = torch.cat([torch.zeros((*points.shape[:-1],1),dtype=torch.int64,device='cuda'),points],dim=-1)
#     embeds = patchloader(points.view(-1,3))
#     return embeds,points

# def get_2d_embeds(image,scale,pipeline,resolution=4):
#     # pyramid = get_clip_pyramid(image,pipeline=pipeline,image_scale=scale)
#     # embeds,points = get_grid_embeds(pyramid,image.shape[0]//resolution,image.shape[1]//resolution,image.shape[0],image.shape[1],scale)
#     patchloader = get_clip_patchloader(image,pipeline=pipeline,image_scale=scale)
#     embeds,points = get_grid_embeds_patch(patchloader,image.shape[0]//resolution,image.shape[1]//resolution,image.shape[0],image.shape[1],scale)
#     return embeds,points

# def get_2d_relevancy(image,scale,pipeline,prompt):
#     embeds,points = get_2d_embeds(image,scale,pipeline=pipeline)
#     pipeline.image_encoder.set_positives([prompt])
#     rel = pipeline.image_encoder.get_relevancy(embeds,0)[:,0:1]
#     rel = rel.view(points.shape[0],points.shape[1],1)
#     return rel

# def get_nerf2world(pipeline):
#     dp_outputs = pipeline.datamanager.train_dataparser_outputs
#     applied_transform = np.eye(4)
#     applied_transform = np.eye(4)
#     applied_transform[:3, :] = dp_outputs.dataparser_transform.numpy() #world to ns
#     applied_transform = np.linalg.inv(applied_transform)
#     applied_transform = applied_transform @ np.diag([1/dp_outputs.dataparser_scale]*3+[1]) #scale is post
#     return torch.tensor(applied_transform,device='cuda').float()

# def add_row(c2w):
#     return torch.cat([c2w,torch.tensor([[0,0,0,1]],device=c2w.device)],dim=0)

# def transform_to_world(camera,pipeline):
#     Hns2world = get_nerf2world(pipeline).cuda()
#     camera.camera_to_worlds = (Hns2world @ add_row(camera.camera_to_worlds))[:3,:]
#     return camera

# def transform_to_ns(camera,pipeline):
#     Hns2world = get_nerf2world(pipeline).cuda()
#     Hworld2ns = torch.inverse(Hns2world)
#     camera.camera_to_worlds = (Hworld2ns @ add_row(camera.camera_to_worlds))[:3,:]
#     return camera

# def transform_cam(camera,H):
#     camera.camera_to_worlds = (add_row(camera.camera_to_worlds) @ H)[:3,:]
#     return camera

# def get_lerf_outputs(camera,pipeline,downscale_res=4,image_coords = None, physical_scale:Optional[float] = None, image_scale:Optional[float] = None):
#     """
#     Renders the outputs from lerf at the camera1 pose inside the pipeline_updated frame
#     To do this we need to multiply by the respective transforms

#     image_scale: scale of image height ratio to set the lerf query scale to
#     physical_scale  
#     """
#     assert not (physical_scale is not None and image_scale is not None), "Cannot provide both physical and image scale"
#     camera.rescale_output_resolution(1/downscale_res)
#     ray_bundle = camera.generate_rays(0,coords=image_coords)
#     ray_bundle.metadata['height']=camera.height.item()
#     ray_bundle.metadata['fy']=camera.fx.item()
#     if physical_scale is not None:
#         ray_bundle.metadata['override_scales']=[physical_scale]
#     if image_scale is not None:
#         ray_bundle.metadata['clip_scales'] = torch.ones((ray_bundle.origins.shape[0],1),device=ray_bundle.origins.device)*image_scale
#     if image_coords is not None:
#         outputs = pipeline.model(ray_bundle)
#     else:
#         outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
#     camera.rescale_output_resolution(downscale_res)
#     return outputs

# def get_2d_dino_embeds(
#     image : torch.Tensor, 
#     dino_model_type : str = "dino_vits8", 
#     dino_stride = 8, 
#     dino_load_size = 500, 
#     dino_layer = 11, 
#     dino_facet = "key", 
#     dino_bin = False,  
#     extractor = None,
# ):
#     if extractor is None:
#         extractor = ViTExtractor(dino_model_type, dino_stride)
#     u_image = torch.permute(image, (2,0,1)) # convert to 3, w, h
#     assert u_image.shape[0] == 3
#     prep = transforms.Compose([
#         transforms.Resize(dino_load_size, antialias=None),
#         transforms.CenterCrop((dino_load_size[0], dino_load_size[1])), 
#         transforms.Normalize(mean=extractor.mean, std=extractor.std)
#     ])
#     u_image = prep(u_image).cuda()
#     print(u_image.shape)
#     with torch.no_grad():
#         descriptors = extractor.extract_descriptors(
#             u_image.unsqueeze(0),
#             [dino_layer],
#             dino_facet,
#             dino_bin,
#         )
#     descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
#     print(descriptors.shape)
#     return descriptors.cpu().detach()

# def dino_pca(dino_feats):
#     dino_feats.norm(dim=-1).max()
#     dino_pca = torch.pca_lowrank(dino_feats.view(-1, dino_feats.shape[-1]), q=3)[0].reshape(dino_feats.shape[0], -1, 3)
#     dino_pca -= dino_pca.reshape(-1, dino_pca.shape[-1]).min(dim=0).values
#     dino_pca /= dino_pca.reshape(-1, dino_pca.shape[-1]).max(dim=0).values
#     return dino_pca

# def normalize(attn : np.ndarray):
#     return (attn - np.min(attn)) / (np.max(attn) - np.min(attn))

# def unnormalize_fn(mean : tuple, std : tuple) -> transforms.Compose:
#     """
#     returns a transformation that turns torch tensor to PIL Image
#     """
#     return transforms.Compose(
#         [
#             transforms.Normalize(
#                 mean=tuple(-m / s for m, s in zip(mean, std)),
#                 std=tuple(1.0 / s for s in std),
#             ),
#             transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)), 
#             transforms.ToPILImage(),
#         ]
#     )

# def get_dino_attn_maps(
#     image : torch.Tensor, 
#     dino_model_type : str = "dino_vits8", 
#     dino_stride = 8, 
#     dino_load_size = (500, 892),
#     crop_ratio = 1,
#     extractor = None,
#     **kwargs,
# ) -> torch.Tensor:
#     """
#     Extracts the attention maps from an image using a specified DINO (DIstillation of NOt only a pretrained supervisor) model.

#     Parameters:
#     -----------
#     image : torch.Tensor
#         Input image tensor. Expected shape is (width, height, 3).

#     dino_model_type : str, optional (default="dino_vits8")
#         The type of DINO model to use for attention map extraction. For example, "dino_vits8" for DINO with Vision Transformer with 8 heads.

#     dino_stride : int, optional (default=8)
#         The stride of the DINO model.

#     dino_load_size : Tuple[int, int], optional (default=(500, 892))
#         The dimensions to which the input image should be resized before feeding into the DINO model.

#     crop_ratio : int, optional (default=1)
#         The ratio to crop the resized image. 1 means no cropping.

#     extractor : Optional[Any], optional (default=None)
#         A custom extractor object that contains methods for attention map extraction. If None, a default extractor based on the `dino_model_type` and `dino_stride` will be used.

#     **kwargs : dict
#         Additional keyword arguments.

#     Returns:
#     --------
#     torch.Tensor
#         Attention maps generated by the DINO model.

#     Example:
#     --------
#     >>> image = torch.rand((128, 128, 3))
#     >>> attn_maps = get_dino_attn_maps(image, dino_model_type="dino_vits8")

#     Notes:
#     ------
#     The function assumes that the input image tensor shape is (width, height, 3) and will be permuted to (3, width, height) for model input.
#     """
#     if extractor is None:
#         extractor = ViTExtractor(dino_model_type, dino_stride)
#     u_image = torch.permute(image, (2,0,1)) # convert to 3, w, h
#     assert u_image.shape[0] == 3
#     prep = transforms.Compose([
#         transforms.Resize(dino_load_size, antialias=None),
#         transforms.CenterCrop((dino_load_size[0] // crop_ratio, dino_load_size[1] // crop_ratio)), 
#         transforms.Normalize(mean=extractor.mean, std=extractor.std)
#     ])
#     u_image = prep(u_image)[None, ... ]
#     with torch.no_grad():
#         attns = extractor.extract_last_attention(u_image)
#     return attns

# def get_diff_attn_maps(
#     original_attns : np.ndarray, 
#     updated_attns : np.ndarray, 
#     thres : float = 0.5, 
#     clean : bool = True,
#     return_attns : bool = False,
# ) -> np.ndarray:
#     """
#     Computes the difference between two attention maps and returns the binary mask highlighting the differences. Optionally, can also return the updated attention map masked by the binary mask.

#     Parameters:
#     -----------
#     original_attns : np.ndarray
#         The original attention maps. Shape should be compatible with `updated_attns`.
    
#     updated_attns : np.ndarray
#         The updated attention maps to be compared with `original_attns`. Shape should be compatible with `original_attns`.
    
#     thres : float, optional (default=0.1)
#         The threshold value used for normalizing the median attention map. Values above this threshold will be considered significant.
    
#     clean : bool, optional (default=True)
#         If True, morphological operations (erosion followed by dilation) are performed on the binary mask to remove noise.
    
#     return_attns : bool, optional (default=True)
#         If True, returns the updated attention maps masked by the binary mask.
#         If False, returns only the binary mask.

#     Returns:
#     --------
#     np.ndarray
#         If `return_attns` is True, returns the updated attention maps masked by the binary mask.
#         If `return_attns` is False, returns only the binary mask highlighting the difference between `original_attns` and `updated_attns`.

#     Example:
#     --------
#     >>> original_attns = torch.rand((5, 128, 128))
#     >>> updated_attns = torch.rand((5, 128, 128))
#     >>> diff_attn_map = get_diff_attn_maps(original_attns, updated_attns, thres=0.2, clean=True, return_attns=False)
#     """
#     assert original_attns.shape == updated_attns.shape, "Attention maps must be the same shape"
#     attns_1 = normalize(np.sum(original_attns, axis=0)) > thres
#     attns_2 = normalize(np.sum(updated_attns, axis=0)) > thres
#     # import pdb; pdb.set_trace()
#     fig, ax = plt.subplots(2)
#     ax[0].imshow(normalize(np.sum(original_attns, axis=0)))
#     ax[1].imshow(normalize(np.sum(updated_attns, axis=0)))
#     plt.show()
#     intersect = attns_1 & attns_2
#     bool_mask_1 = attns_1 ^ intersect
#     bool_mask_2 = attns_2 ^ intersect
#     bool_mask = bool_mask_1 | bool_mask_2
    
#     # fig, ax = plt.subplots(6)
#     # ax[0].imshow(attns_1)
#     # ax[1].imshow(attns_2)
#     # ax[2].imshow(intersect)
#     # ax[4].imshow(bool_mask_1)
#     # ax[3].imshow(bool_mask_2)
#     # ax[5].imshow(bool_mask)
#     # plt.show()
#     original = bool_mask.copy()
#     # import pdb; pdb.set_trace()
    
#     erosion_kernel_size = 5
#     dialation_kernel_size = 70
#     if clean:
#         # Convert boolean mask to binary image
#         binary_mask = (original.astype(np.uint8) * 255)

#         # Define kernel for morphological operations (you can adjust the size)
#         erosion_kernel = np.ones((erosion_kernel_size,erosion_kernel_size),np.uint8)
#         dilation_kernel = np.ones((dialation_kernel_size,dialation_kernel_size),np.uint8)

#         # Use erosion followed by dilation
#         erosion = cv2.erode(binary_mask, erosion_kernel, iterations=3)
#         cleaned_mask = cv2.dilate(erosion, dilation_kernel, iterations=1)
        
#         bool_mask = cleaned_mask.astype(bool)
    
#     # fig, ax = plt.subplots(9)
#     # ax[0].imshow(attns_1)
#     # ax[1].imshow(attns_2)
#     # ax[2].imshow(intersect)
#     # ax[3].imshow(bool_mask_1)
#     # ax[4].imshow(bool_mask_2)
#     # ax[5].imshow(original)
#     # ax[6].imshow(bool_mask)
#     # ax[7].imshow(normalize(np.sum(original_attns, axis=0)))
#     # ax[8].imshow(normalize(np.sum(updated_attns, axis=0)))
#     # plt.show()

#     # import pdb; pdb.set_trace()

#     if return_attns: 
#         raise NotImplementedError
#         # return bool_mask * updated_attns
#     else:
#         return bool_mask