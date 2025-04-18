import open3d as o3d
import numpy as np
import trimesh
import coacd
from pathlib import Path
from copy import deepcopy
import munch
import scipy.spatial


def print_mesh_info(info):
    """Print the mesh information dictionary."""
    print("Mesh Information:")
    print("Volume of the mesh:", info["volume"])
    print('number of vertices: ',
          info["vertices"].shape[0], 'number of faces: ', info["faces"].shape[0])
    print("AABB Min Bound:", info["aabb"].get_min_bound())
    print("AABB Max Bound:", info["aabb"].get_max_bound())
    print("AABB Extents:", info["aabb"].get_extent())
    print("AABB Volume:", info["aabb"].volume())
    print("OBB Min Bound:", info["obb"].get_min_bound())
    print("OBB Max Bound:", info["obb"].get_max_bound())
    print("OBB Volume:", info["obb"].volume())
    print("OBB Rotation:\n", info["obb"].R)
    print("OBB Extents:", info["obb_extents"])


def load_mesh(file_path,  vis=False):
    """Load a mesh file using Open3D and Trimesh and calculate its volume and bounding boxes.

    Args:
        file_path (Path): Path to mesh file
        vis (bool, optional): visualization option. Defaults to False.

    Returns:
        mesh in open3d and trimesh format, info (dict): Dictionary containing information about object.
    """
    info = {}
    mesh_o3d = o3d.io.read_triangle_mesh(file_path)
    mesh_trimesh = trimesh.load(file_path)

    # trimesh can calculate the volume of the mesh if it is watertight
    if mesh_trimesh.is_watertight:
        # Calculate the volume
        volume = mesh_trimesh.volume
        info["volume"] = volume
    else:
        info["volume"] = "undefined"

    # get vertices and faces and covert then to numpy arrays
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    info["vertices"] = vertices
    info["faces"] = faces

    # create a new mesh from the vertices and faces to remove all additional information like color or normals
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Calculate the axis-aligned bounding box
    aabb = mesh_o3d.get_axis_aligned_bounding_box()
    info["aabb"] = aabb

    # Compute the Oriented Bounding Box (OBB)
    obb = mesh_o3d.get_oriented_bounding_box()
    info["obb"] = obb

    # simple way to get the size of the obb is to rotate them to the axis-aligned bounding box and then get the extents
    obb_rotated = deepcopy(obb)
    obb_rotated.rotate(obb_rotated.R.T)
    # print("OBB Extents:", obb_rotated.get_max_bound()-obb_rotated.get_min_bound())
    info["obb_extents"] = obb_rotated.get_max_bound() - \
        obb_rotated.get_min_bound()
    # simple visualization of the bounding boxes and the mesh
    if vis:
        aabb.color = (1, 0, 0)  # RGB values in the range [0, 1]
        obb.color = (0, 0, 1)
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries(
            [mesh_o3d, aabb, obb], mesh_show_wireframe=True)

    return mesh, mesh_trimesh, info


def scale_mesh(mesh, scale_factor):
    # Create a scaling transformation matrix
    scale_matrix = np.eye(4)  # 4x4 identity matrix
    scale_matrix[:3, :3] *= scale_factor  # Apply scaling to the diagonal

    # Apply the scaling transformation to the mesh
    mesh.transform(scale_matrix)

def save_vf_as_ply(v, f, dir_path, name, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(dir_path/(name+".ply"), mesh)

def save_vf_as_obj(v,f, dir_path, name):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d.io.write_triangle_mesh(dir_path/(name+".obj"), mesh)
