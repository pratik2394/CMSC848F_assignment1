import argparse
import torch
import imageio
from PIL import Image
import numpy as np
import pytorch3d
from starter.utils import get_device, get_points_renderer, unproject_depth_image
from starter.render_generic import load_rgbd_data

def render_point_cloud(image_size=256, background_color=(1, 1, 1), device=None, num_frames = 20, data_path = "data/rgbd_data.pkl"):

    rgbd_data = load_rgbd_data(path=data_path)

    points_1, rgb_1  = unproject_depth_image(image=torch.tensor(rgbd_data['rgb1']), mask=torch.tensor(rgbd_data['mask1']),
                                                depth=torch.tensor(rgbd_data['depth1']), camera=rgbd_data['cameras1'])

    device = get_device()
    renderer = get_points_renderer(image_size = image_size, background_color=background_color)

    point_cloud1 = pytorch3d.structures.Pointclouds(points=points_1.to(device).unsqueeze(0), features=rgb_1.to(device).unsqueeze(0))

    my_images = []
    for i in range(num_frames):
        dist = 3.0
        elev = 0
        azim = 360 * i / num_frames
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud1, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))
    imageio.mimsave("output/point_cloud1.gif", my_images, fps=10)


    points_2, rgb_2 = unproject_depth_image(image=torch.tensor(rgbd_data['rgb2']), mask=torch.tensor(rgbd_data['mask2']), 
                                                depth=torch.tensor(rgbd_data['depth2']), camera=rgbd_data['cameras2'])

    point_cloud2 = pytorch3d.structures.Pointclouds(points=points_2.to(device).unsqueeze(0), features=rgb_2.to(device).unsqueeze(0))

    my_images = []
    for i in range(num_frames):
        dist = 3.0
        elev = 0
        azim = 360 * i / 20
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud2, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))
    imageio.mimsave("output/point_cloud2.gif", my_images, fps=10)

    rotated_all_points = torch.cat([points_1, points_2], dim=0)
    all_colors = torch.cat([torch.tensor(rgb_1), torch.tensor(rgb_2)], dim=0)
    point_cloud_combined = pytorch3d.structures.Pointclouds(points=rotated_all_points.to(device).unsqueeze(0), features=all_colors.to(device).unsqueeze(0))

    my_images = []
    for i in range(num_frames):
        dist = 3.0
        elev = 0
        azim = 360 * i / num_frames
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud_combined, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))
    imageio.mimsave("output/point_cloud_combined.gif", my_images, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    render_point_cloud(
        image_size=args.image_size,
        num_frames=args.num_frames,
        data_path = "data/rgbd_data.pkl",
        background_color=(1, 1, 1)
    )
