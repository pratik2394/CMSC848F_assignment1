import argparse
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image

from starter.render_generic import load_rgbd_data


def render_torus(image_size=256, num_samples=200, device=None):
    """
    torus equations:
    x = (R + r cos(alpha)) cos(theta)
    y = (R + r cos(alpha)) sin(theta)
    z = r sin(alpha)
    """

    if device is None:
        device = get_device()
    R = 1.0
    r = 0.25
    alpha = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(theta, alpha)

    x = (torch.tensor(R)+torch.tensor(r)*torch.cos(alpha)) * torch.cos(theta)
    y = (torch.tensor(R)+torch.tensor(r)*torch.cos(alpha)) * torch.sin(theta)
    z = torch.tensor(r) * torch.sin(alpha) 

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)

    

    my_images = []
    for i in range(20):
        dist = 3.0
        elev = 0
        azim = 360 * i / 20
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(torus_point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))

    imageio.mimsave("output/5_2_torus_360.gif", my_images, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    render_torus(
        image_size=args.image_size,
        num_samples=args.num_samples,
    )
