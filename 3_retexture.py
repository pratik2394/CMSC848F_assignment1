"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def retexture_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    color1 = [0.0, 0.0, 1.0]
    color2 = [1.0, 0.0, 0.0]
    min_z = vertices[:,:,2].min()
    max_z = vertices[:,:,2].max()
    gradient = (vertices[:, :, 2] - min_z) / (max_z - min_z)
    textures = gradient[:, :, None] * torch.tensor(color2) + (1 - gradient[:, :, None]) * torch.tensor(color1)
    

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    my_images = []
    for i in range(20):
        dist = 3.0
        elev = 0
        azim = 360 * i / 20
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))
    imageio.mimsave("output/cow_retextured360.gif", my_images, fps=10)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.707, -0.707], [0.0, 0.707, 0.707]]).unsqueeze(0), T=torch.tensor([[0, 0, 4]]), fov=60, device=device
    )

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="output/cow_retextured.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = retexture_cow(cow_path=args.cow_path, image_size=args.image_size)
    # plt.imsave(args.output_path, image)
