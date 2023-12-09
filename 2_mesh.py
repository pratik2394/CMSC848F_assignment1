
import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import os
import numpy as np
from PIL import Image
import imageio
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer

def construct_tetrahedron(image_size=256, color=[0.7, 0.7, 1], device=None, output_file="output/tetrahedron_mesh.gif"):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # tetrahedron.
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0)  
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]).unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    my_images = []
    for i in range(20):
        dist = 3.0
        elev = 2
        azim = 360 * i / 20
        R, T = pytorch3d.renderer.look_at_view_transform(dist, elev, azim, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        frame = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(frame))

    imageio.mimsave("output/tetrahedron_mesh.gif", my_images, fps=10)

def construct_cube(image_size=256, color=[0.7, 0.7, 1], device=None, output_file="output/cube_mesh.gif"):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Define vertices and faces for a tetrahedron.
    vertices = torch.tensor([ [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 
                              [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]).unsqueeze(0) 

    faces = torch.tensor([  [0, 1, 2],  [0, 2, 3],  
                            [0, 1, 4],  [0, 4, 7],       
                            [0, 3, 6],  [0, 6, 7],  
                            [4, 5, 7],  [6, 5, 7],       
                            [1, 2, 5],  [1, 5, 4],  
                            [2, 3, 5],  [3, 5, 6]]).unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

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

    imageio.mimsave(output_file, my_images, fps=10)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    construct_tetrahedron(image_size=args.image_size, output_file="output/tetrahedron_mesh.gif")
    construct_cube(image_size=args.image_size, output_file="output/cube_mesh.gif")
