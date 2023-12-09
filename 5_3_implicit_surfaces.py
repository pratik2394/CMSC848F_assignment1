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

def render_torus_mesh(image_size=256, voxel_size=64, device=None):

    """ 
    Torus equation in implicit form: x^2 + y^2 + z^2 + (R^2 - r^2) - 4R^2(x^2 + y^2) = 0
    """

    if device is None:
        device = get_device()
    min_value = -2.5
    max_value = 2.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    R = 1.0
    r = 0.25

    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    
    gradient = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    colors = gradient[:, 2][:, None] * torch.tensor([1.0, 0.0, 0.0]) + (1 - gradient[:, 2][:, None]) * torch.tensor([0.0, 0.0, 1.0])

    textures = pytorch3d.renderer.TexturesVertex(colors[None, ...])

    torus_mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

 
    my_images = []
    for i in range(20):
        dist = 3.0
        elev = 0
        azim = 360 * i / 20
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(torus_mesh, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))

    imageio.mimsave("output/5_3_torus_mesh_360.gif", my_images, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    render_torus_mesh(
        image_size=args.image_size,
        voxel_size=args.voxel_size,
    )
