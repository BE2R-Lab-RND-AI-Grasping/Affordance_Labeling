import open3d as o3d
import numpy as np
import trimesh
import coacd
from pathlib import Path
from copy import deepcopy

def create_default_convex_decomposition(mesh:o3d.geometry.TriangleMesh, data_dir:Path,  vis:bool=False):
    #  coacd expects the vertices and faces in the form of np.array, they can be obtained from the trimesh representation directly, or from the open3d with numpy
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    mesh = coacd.Mesh(vertices, faces)
    parts = coacd.run_coacd(mesh) # a list of convex hulls.
    # each part is a list which contains two lists: vertices and faces

    new_mesh = o3d.geometry.TriangleMesh()
    if vis: new_mesh_vis = o3d.geometry.TriangleMesh()

    n = 0
    for vs, fs in parts:
        # for each part in decomposition create an open3d TriangleMesh object and save it to the data_dir
        current_mesh = o3d.geometry.TriangleMesh()
        # Assign vertices and faces to the mesh
        current_mesh.vertices = o3d.utility.Vector3dVector(vs)
        current_mesh.triangles = o3d.utility.Vector3iVector(fs)
        o3d.io.write_triangle_mesh(data_dir/f"object_part_{n}.obj", current_mesh)
        # if visualization is enabled create another colored mesh with random color
        if vis:
            current_mesh_vis = o3d.geometry.TriangleMesh()
            current_mesh_vis.vertices = o3d.utility.Vector3dVector(vs)
            current_mesh_vis.triangles = o3d.utility.Vector3iVector(fs)
            current_mesh_vis.paint_uniform_color(np.random.rand(3))
            new_mesh_vis+=current_mesh_vis
        n+=1
        # open 3d meshes can be added together to be merged into one mesh 
        new_mesh+=current_mesh

    # Get the vertices as a NumPy array
    vertices = np.asarray(new_mesh.vertices)
    # Get the triangle faces (indices of vertices)
    faces = np.asarray(new_mesh.triangles)
    print('number of vertices: ', vertices.shape[0], 'number of faces: ', faces.shape[0])
    # new_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(data_dir/"object_convex_decomposition.obj", new_mesh)
    if vis:
        o3d.visualization.draw_geometries([new_mesh_vis])

    # Check for duplicate vertices
    unique_vertices, indices, counts = np.unique(vertices, axis=0, return_index=True, return_counts=True)

    # Find duplicate vertices
    duplicate_indices = np.where(counts > 1)[0]

    # Print results
    if len(duplicate_indices) > 0:
        print(f"Found {len(duplicate_indices)} duplicate vertices.")
        for idx in duplicate_indices:
            print(f"Vertex {unique_vertices[idx]} appears {counts[idx]} times.")
    else:
        print("No duplicate vertices found.")
    return new_mesh
