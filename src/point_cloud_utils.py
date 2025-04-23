import open3d as o3d
import numpy as np
import trimesh
import coacd
from pathlib import Path
from copy import deepcopy
import munch
import scipy.spatial


def create_point_cloud(mesh: o3d.geometry.TriangleMesh, num_points: int, method: str = "uniform", vis: bool = False):
    """Create a point cloud from a mesh and return it as an Open3D point cloud object.

    Args:
        mesh (o3d.geometry.TriangleMesh): mesh of the object
        num_points (int): desired number of points in the point cloud
        method (str, optional): method of sampling. Defaults to "uniform".
        vis (bool, optional): request for visualization. Defaults to False.

    Returns:
        open3d PointCloud: point cloud of the object
    """
    if method == "uniform":
        point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    elif method == "poisson":
        point_cloud = mesh.sample_points_poisson_disk(
            number_of_points=num_points)
    if vis:
        o3d.visualization.draw_geometries([point_cloud])
    pc_colorless = o3d.geometry.PointCloud()
    pc_colorless.points = point_cloud.points
    return point_cloud, pc_colorless


def calculate_signed_distances(point_cloud: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh):
    # Compute vertex normals (required for signed distance)
    mesh.compute_vertex_normals()
    # convert the point cloud to a numpy array with a particular dtype what is required by the tensor-based raycasting scene
    point_cloud_array = np.asarray(point_cloud.points, dtype=np.float32)
    # Convert the mesh to a tensor-based TriangleMesh
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    # Add the mesh to the scene
    mesh_id = scene.add_triangles(tensor_mesh)
    # Compute the signed distances for the entire point cloud
    signed_distances = scene.compute_signed_distance(point_cloud_array)
    # Convert the signed distances to a NumPy array
    signed_distances = signed_distances.numpy()
    return signed_distances


def to_numpy_labeled(color_labeled_pc: o3d.geometry.PointCloud, color_to_label):
    """Convert a labeled point cloud to a NumPy.

    Args:
        color_labeled_pc (o3d.geometry.PointCloud): labeled point cloud
        color_map (munch.Munch): color map for labels

    Returns:
        np.ndarray: labeled point cloud as a NumPy array
    """
    if not color_labeled_pc.has_colors():
        raise ValueError("Mesh does not have vertex colors.")

    # Extract vertex colors and points
    vertex_colors = (np.asarray(color_labeled_pc.colors)
                     * 255).astype(np.int32)
    labels = np.array([[color_to_label(color)] for color in vertex_colors])
    vertices = np.asarray(color_labeled_pc.points)
    return np.hstack((vertices, labels))


def get_single_color_pcs(color_labeled_pc: o3d.geometry.PointCloud):
    """Extracts unique colors from a colored point cloud and creates separate point clouds for each color.

    Args:
        color_labeled_pc (o3d.geometry.PointCloud): labeled point cloud
    """
    # Check if the mesh has vertex colors
    if not color_labeled_pc.has_colors():
        print("No vertex colors found in the mesh.")
        exit()

    # Extract vertex colors and points
    vertex_colors = np.asarray(color_labeled_pc.colors)
    vertices = np.asarray(color_labeled_pc.points)

    # Find unique colors (rounded to avoid floating-point issues)
    # Here we round to 3 decimal places to group similar colors
    rounded_colors = (vertex_colors*255).astype(np.int32)

    # Get unique colors
    unique_colors = np.unique(rounded_colors, axis=0)

    # Process each unique color
    pcs = []
    colors_list = []
    for _, color in enumerate(unique_colors):
        # Find the vertices that match the current color
        color_mask = np.all(rounded_colors == color, axis=1)
        filtered_vertices = vertices[color_mask]
        # Create a new PointCloud object for the vertices of this color
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_vertices)
        colors = np.tile(color, (len(filtered_vertices), 1))
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        pcs.append(point_cloud)
        colors_list.append((color*255).astype(np.int32))

    return pcs,  colors_list

# def create_pc_from_masked_depth(depth_image_path, mask):
#     depth = o3d.io.read_image(depth_image_path)
#     o3d.geometry.PointCloud.create_from_depth_image()

from src import object_render
def rotate_back(direction, angle, pcd:o3d.geometry.PointCloud):
    rot_to_z = object_render.get_rotation_to_z(direction)
    rot_around_z = object_render.get_rotation_around_z(angle)

    pcd.rotate(np.linalg.inv(rot_around_z), center=(0, 0, 0))
    pcd.rotate(np.linalg.inv(rot_to_z), center=(0, 0, 0))


def create_masked_depths(depth_dir):
    n=0
    while Path(depth_dir/(f"depth_{n}.png")).exists():
        file_name = depth_dir/(f"depth_{n}.png")
        depth_mask_file_name = depth_dir/(f"depth_mask_{n}.npy")
        depth = o3d.io.read_image(file_name)
        depth_mask = np.load(depth_mask_file_name)
        mask = depth_mask.astype("bool")
        depth_array = np.asarray(depth)
        masked_depth_array = depth_array * mask
        o3d.io.write_image(depth_dir/(f"masked_depth_{n}.png"),o3d.geometry.Image(masked_depth_array))
        n+=1
    
def create_pc_from_masked_depth(depth_image_masked, direction, rot, camera_params):
    labeling_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_masked, depth_scale=10, intrinsic=camera_params.intrinsic, extrinsic=camera_params.extrinsic, depth_trunc=1000.0)
    labeling_pcd.paint_uniform_color([1, 0, 0])
    # labeling_pcd.paint_uniform_color(np.random.rand(3))
    rotate_back(direction, rot, labeling_pcd)
    return labeling_pcd

def process_masked_depths(depth_dir, directions, idx_list,camera_params):
    n=0
    result_pcd = o3d.geometry.PointCloud()
    result_pcds = [] 
    while Path(depth_dir/(f"masked_depth_{n}.png")).exists():
        file_name = depth_dir/(f"masked_depth_{n}.png")
        depth_image_masked = o3d.io.read_image(file_name)
        max_ind = idx_list[n]
        direction, rot = directions[max_ind][:-1], directions[max_ind][-1]
        pcd = create_pc_from_masked_depth(depth_image_masked, direction, rot, camera_params)
        result_pcd += pcd
        result_pcds.append(pcd)
        n+=1
    return result_pcd, result_pcds


