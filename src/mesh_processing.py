"""Functions for mesh processing for the affordance labeling task."""
from pathlib import Path
import open3d as o3d
from src.model_save_load import load_mesh
from src.convex_decomposition import create_default_convex_decomposition
from src.point_cloud_utils import create_point_cloud

def process_mesh(path_to_dir:Path):
    files = path_to_dir.glob("initial_mesh.*")
    if not files:
        raise ValueError("No mesh file found in the directory.")
    mesh_file = next(path_to_dir.glob("initial*"))
    mesh, _, _ = load_mesh(mesh_file, vis=False)
    # first step is convex decomposition of the mesh with loading each part into the mesh directory
    # the result is the new mesh that merges all parts and it is saved in the directory
    # decomposition can take a long time, the result is saved in the data_dir in the obj format
    if (path_to_dir/"object_convex_decomposition.obj").exists():
        new_mesh = o3d.io.read_triangle_mesh(path_to_dir/"object_convex_decomposition.obj")
    else:
        new_mesh = create_default_convex_decomposition(mesh, path_to_dir, vis=False)
    
    # second step is to create a colorless point cloud from the new mesh.
    # There is a known problem that the convex parts have faces and vertices on the boundaries that are connected to the other parts.
    # As a result there are faces and vertices that are inside the merged object. And the points are spawned inside the object.
    if not (path_to_dir/"points_decomposed.ply").exists():
        point_cloud, point_cloud_colorless=create_point_cloud(new_mesh, 10000, vis=False)
        o3d.io.write_point_cloud(path_to_dir/"points_decomposed.ply", point_cloud_colorless, write_ascii=True)





