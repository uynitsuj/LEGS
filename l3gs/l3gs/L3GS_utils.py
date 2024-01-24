import torch

class Utils:
    # @profile
    def deproject_to_RGB_point_cloud(image, depth_image, camera, scale, num_samples = 250, device = 'cuda:0'):
        """
        Converts a depth image into a point cloud in world space using a Camera object.
        """
        
        # import pdb; pdb.set_trace()
        # c2w = camera.camera_to_worlds.cpu()
        # depth_image = depth_image.cpu()
        # image = image.cpu()
        c2w = camera.camera_to_worlds.to(device)
        if len(c2w.shape) == 3:
            c2w = c2w.squeeze(0)
        depth_image = depth_image.to(device)
        image = image.to(device)
        fx = camera.fx.item()
        fy = camera.fy.item()
        # cx = camera.cx.item()
        # cy = camera.cy.item()

        _, _, height, width = depth_image.shape

        grid_x, grid_y = torch.meshgrid(torch.arange(width, device = device), torch.arange(height, device = device), indexing='ij')
        grid_x = grid_x.transpose(0,1).float()
        grid_y = grid_y.transpose(0,1).float()

        flat_grid_x = grid_x.reshape(-1)
        flat_grid_y = grid_y.reshape(-1)
        flat_depth = depth_image[0, 0].reshape(-1)
        flat_image = image.reshape(-1, 3)

        ### simple uniform sampling approach
        num_points = flat_depth.shape[0]
        sampled_indices = torch.randint(0, num_points, (num_samples,))

        sampled_depth = flat_depth[sampled_indices] * scale
        # sampled_depth = flat_depth[sampled_indices]
        sampled_grid_x = flat_grid_x[sampled_indices]
        sampled_grid_y = flat_grid_y[sampled_indices]
        sampled_image = flat_image[sampled_indices]

        X_camera = (sampled_grid_x - width/2) * sampled_depth / fx
        Y_camera = -(sampled_grid_y - height/2) * sampled_depth / fy

        ones = torch.ones_like(sampled_depth)
        P_camera = torch.stack([X_camera, Y_camera, -sampled_depth, ones], dim=1)
        
        homogenizing_row = torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype, device=device)
        camera_to_world_homogenized = torch.cat((c2w, homogenizing_row), dim=0)

        P_world = torch.matmul(camera_to_world_homogenized, P_camera.T).T
        
        return P_world[:, :3], sampled_image
