'''
Using Kaoling to convert from SDF to mesh and mesh to SDF differentiably.
'''

import copy
import importlib
import inspect
import logging
import os
from typing import List, Optional, Union
import sys
sys.path.append('./Hunyuan3D-2')

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from pytorch3d.structures import Meshes, join_meshes_as_scene

import kaolin.non_commercial as knc
import kaolin.ops.mesh as mesh_ops
import kaolin.metrics as km
from pytorch3d.io import IO
from pytorch3d.transforms import quaternion_to_matrix, Rotate, Translate


def generate_dense_grid_points(bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               octree_depth: int,
                               indexing: str = "ij",
                               octree_resolution: int = None,
                               ):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_depth)
    if octree_resolution is not None:
        num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def remove_degenerate_faces(mesh: Meshes, eps: float = 1e-9) -> Meshes:
    """
    Drops any face whose area is < eps.
    """

    # TODO: needs debugging, starting from V[F] part as it couldn't be indexed with that tensor

    V = mesh.verts_packed()     
    F = mesh.verts_packed()    

    tris = V[F] # 3D coords of each triangle
    v0, v1, v2 = tris.unbind(1)  # each is (F,3)

    # compute half the cross‐product norm = area
    e1 = v1 - v0
    e2 = v2 - v0
    cross = torch.cross(e1, e2, dim=1)
    area2 = torch.sum(cross**2, dim=1)  # (F,)  squared area*4

    keep = area2 > (2 * eps) ** 2       # squared threshold for robustness

    # rebuild the mesh
    new_faces = F[keep]
    return Meshes(verts=[V], faces=[new_faces]).to(mesh.device)


def sdf2mesh(sdf, grid_points=None, device='cuda', resolution=64):
    '''
    Using Kaolin-Flexicubes to convert from sdf to mesh differentiably.
    sdf: flattened sdf of shape (resolution^3, ), -1 inside and 1 outside
    grid_points: if not None, use this as the grid points instead of the voxel vertices from flexicubes' voxel grid
    returns pytorch3d mesh
    '''
    flexi = knc.FlexiCubes(device='cuda')
    _, cube_indices = flexi.construct_voxel_grid(resolution)  # grid setup
    verts, faces, _ = flexi(grid_points, sdf, cube_indices, resolution)
    mesh = Meshes(verts=[verts], faces=[faces], textures=None)
    return mesh


def mesh2sdf(mesh, grid_points=None, device='cuda', resolution=64):
    '''
    Using Kaolin-Flexicubes to convert from mesh to sdf differentiably.
    mesh: pytorch3d mesh
    grid_points: if not None, use this as the grid points instead of the voxel vertices from flexicubes' voxel grid
    returns sdf of shape (resolution^3, )
    '''
    mesh = mesh.clone()

    vertices = mesh.verts_padded() 
    faces = mesh.faces_padded()
    
    face_vertices = mesh_ops.index_vertices_by_faces(vertices, faces[0])
    squared_distance, _, _ = km.trianglemesh.point_to_mesh_distance(grid_points.unsqueeze(0), face_vertices)
    distance = torch.sqrt(squared_distance)

    sign = mesh_ops.check_sign(vertices, faces[0], grid_points.unsqueeze(0))
    sign_num = torch.where(sign, torch.tensor(-1.0).to(device), torch.tensor(1.0).to(device))
    sdf = distance * sign_num
    sdf = sdf.squeeze(0)

    return sdf


def test_conversion_mesh2sdf(mesh_path, xyz_samples, save_path, device='cuda', resolution=64):
    '''
    Test the conversion from mesh to sdf and back to mesh
    mesh_path: path to the mesh file
    xyz_samples: the xyz samples to use for the conversion
    '''
    # Load a pytorch3d mesh
    pytorch3d_mesh = IO().load_mesh(mesh_path).to(device)

    # Convert to sdf
    sdf = mesh2sdf(pytorch3d_mesh, xyz_samples, resolution=resolution)

    # Convert back to mesh
    converted_mesh = sdf2mesh(sdf, xyz_samples, resolution=resolution)

    # Save the converted mesh
    IO().save_mesh(converted_mesh, save_path)


def get_sdf_of_meshes(mesh1, mesh2, device, resolution=64):
    '''
    Get the sdf of 2 meshes that share the same grid
    mesh1: pytorch3d mesh
    mesh2: pytorch3d mesh
    returns sdf of shape (resolution^3, ) of the corresponding meshes
    '''
    bbox_min_1 = mesh1.verts_padded().min(dim=1)[0].detach()
    bbox_max_1 = mesh1.verts_padded().max(dim=1)[0].detach()
    bbox_min_2 = mesh2.verts_padded().min(dim=1)[0].detach()
    bbox_max_2 = mesh2.verts_padded().max(dim=1)[0].detach()

    bbox_min = torch.min(bbox_min_1, bbox_min_2)[0]
    bbox_max = torch.max(bbox_max_1, bbox_max_2)[0]

    octree_res = resolution  # or higher for quality
    grid_points, grid_size, _ = generate_dense_grid_points(
        bbox_min=bbox_min.cpu().numpy(),
        bbox_max=bbox_max.cpu().numpy(),
        octree_depth=5,
        octree_resolution=octree_res,
        indexing="ij"
    )
    grid_points = torch.FloatTensor(grid_points).to(device)

    sdf1 = mesh2sdf(mesh1, grid_points, device=device, resolution=resolution)
    sdf2 = mesh2sdf(mesh2, grid_points, device=device, resolution=resolution)

    del grid_points
    return sdf1, sdf2 #, grid_points # NOTE: added grid_points for debugging
