import os
import shutil
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from copy import deepcopy
from src import bbox_utils
from src import object_detection


def get_rotation_to_z(source_vector):
    source_vector = source_vector / np.linalg.norm(source_vector)  # Normalize

    # If already aligned with z-axis, return identity
    if np.allclose(source_vector, [0, 0, 1]):
        return np.eye(3)

    # If opposite to z-axis, rotate by pi around any perpendicular axis (e.g., x-axis)
    if np.allclose(source_vector, [0, 0, -1]):
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
    # Find rotation axis (cross product with z-axis)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(source_vector, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize

    # Find rotation angle (dot product)
    cos_theta = np.dot(source_vector, z_axis)
    theta = np.arccos(cos_theta)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - cos_theta) * (K @ K)

    return R


def get_rotation_around_z(angle_degrees):
    """
    Computes a rotation matrix around the z-axis by a given angle.

    Args:
        angle_degrees (float): Rotation angle in degrees.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])


def get_render(direction_vector, angle, input_mesh: o3d.geometry.TriangleMesh, dir=Path("./"), render_numeration = None):
    # TODO: add the distance to the function parameters
    # create a directory if it does not exist
    if not dir.exists():
        dir.mkdir(parents=True)

    mesh = deepcopy(input_mesh)
    # rotate the mesh according in a way that the input vector aligns with z-axis
    rot_to_z = get_rotation_to_z(direction_vector)
    mesh.rotate(rot_to_z)
    # rotate around z axis
    rot_around_z = get_rotation_around_z(angle)
    mesh.rotate(rot_around_z)

    # Create a visualizer object

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # default width = 1920, height=1080

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    # Set up camera parameters
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    # intrinsic = params.intrinsic
    extrinsic = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 500],
        [0, 0, 0, 1]
    ])
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)

    # Render the scene and capture the image
    vis.poll_events()
    vis.update_renderer()
    if render_numeration is  None:
        render_path = dir / \
            (f"render_{direction_vector[0]: .2f}_{direction_vector[1]: .2f}_{direction_vector[2]: .2f}_{angle}.png")
    else:
        render_path = dir/("render_" + str(render_numeration) + ".png")
    
    vis.capture_depth_image(render_path, do_render=True)

    # depth_path = dir / \
    #     (f"depth_{direction_vector[0]: .2f}_{direction_vector[1]: .2f}_{direction_vector[2]: .2f}_{angle}.png")
    # vis.capture_depth_image(depth_path, do_render=True, depth_scale=10)

    params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    return render_path, params.extrinsic, params.intrinsic


def get_depth_render(direction_vector, angle, input_mesh: o3d.geometry.TriangleMesh, dir=Path("./"), render_numeration = None):
    # TODO: add the distance to the function parameters
    # create a directory if it does not exist
    if not dir.exists():
        dir.mkdir(parents=True)

    mesh = deepcopy(input_mesh)
    # rotate the mesh according in a way that the input vector aligns with z-axis
    rot_to_z = get_rotation_to_z(direction_vector)
    mesh.rotate(rot_to_z)
    # rotate around z axis
    rot_around_z = get_rotation_around_z(angle)
    mesh.rotate(rot_around_z)

    # Create a visualizer object

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # default width = 1920, height=1080

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    # Set up camera parameters
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    # intrinsic = params.intrinsic
    extrinsic = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 500],
        [0, 0, 0, 1]
    ])
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)

    # Render the scene and capture the image
    vis.poll_events()
    vis.update_renderer()
    if render_numeration is  None:
        depth_path = dir / \
            (f"depth_{direction_vector[0]: .2f}_{direction_vector[1]: .2f}_{direction_vector[2]: .2f}_{angle}.png")
    else:
        depth_path = dir/("depth_" + str(render_numeration) + ".png")
    vis.capture_depth_image(depth_path, do_render=True, depth_scale=10)

    params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.destroy_window()

    return depth_path, params.extrinsic, params.intrinsic


def comparison_score(template: np.ndarray, target: np.ndarray, mask_color=np.array([200, 200, 200]), vis=False) -> float:
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_height, target_width = template.shape[:2]
    target = cv2.resize(
        target,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST)
    mask = np.all(target == mask_color, axis=-1)

    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # Invert threshold: detect gray object on white background
    # Threshold the image to separate foreground from background
    _, thresh_target = cv2.threshold(
        gray_target, 240, 255, cv2.THRESH_BINARY_INV)
    _, thresh_template = cv2.threshold(
        gray_template, 100, 255, cv2.THRESH_BINARY)
    thresh_target[mask] = 0
    thresh_template[mask] = 0
    if vis:
        plt.imshow(thresh_template)
        plt.show()
        plt.imshow(thresh_target)
        plt.show()

    return np.logical_and(thresh_target, thresh_template).sum()/np.logical_or(thresh_target, thresh_template).sum(), mask


def get_multiview_renders(mesh, render_dir: Path):
    # create angle samples for all renders
    # TODO: add the function parameters to control the sampling
    azimuth_samples = np.linspace(0, 360, 20, endpoint=False)
    elevation_cos = np.linspace(-1, 1, 10, endpoint=True)
    elevation_samples = (np.rad2deg(np.arccos(elevation_cos))).astype("int32")
    rotation_samples = np.linspace(0, 360, 20, endpoint=False)
    directions = []
    render_numeration = 0
    path_list = []
    for elevation in elevation_samples:
        # if the elevation is 0 or 180, we only need to rotate around the z-axis (spherical coordinates are degenerate in this points)
        if elevation == 0 or elevation == 180:
            for rot in rotation_samples:
                direction = [0, 0, np.cos(np.radians(elevation))]
                path, _, _ = get_render(direction, rot, mesh, render_dir, render_numeration)
                direction.append(rot)
                directions.append(direction)
                path_list.append(path)
                render_numeration += 1
        else:
            for azimuth in azimuth_samples:
                for rot in rotation_samples:
                    direction = [np.cos(np.radians(azimuth))*np.sin(np.radians(elevation)), np.sin(np.radians(azimuth))*np.sin(np.radians(elevation)), np.cos(np.radians(elevation))]
                    path, _, _ = get_render(direction, rot, mesh, render_dir, render_numeration)
                    direction.append(rot)
                    directions.append(direction)
                    path_list.append(path)
                    render_numeration += 1
    # render_iter = render_dir.glob("render_*")
    # path_list = list(render_iter)
    # print(len(path_list))
    np.save(render_dir/"directions.npy", np.array(directions))
    return path_list, directions


def get_best_render_match(path_list, scene_results):
    hand_bbox = scene_results["hand_bbox"]
    object_mask = scene_results["image"]
    image = bbox_utils.add_bbox_to_image(object_mask, hand_bbox)
    bbox = bbox_utils.get_total_bounding_box(image, background_white=True)
    croped_image_scene = bbox_utils.crop_bbox_cv2(image, bbox)
    max_score = -np.inf
    max_ind = -1
    max_mask = None
    cropped = None
    max_bbox = None
    best_image = None
    for i, path in enumerate(path_list):
        image = cv2.imread(path)
        bbox = object_detection.detect_object_with_opencv(image, vis=False)
        croped_image = bbox_utils.crop_bbox_cv2(image, bbox)
        score, mask = comparison_score(croped_image, croped_image_scene)
        if score > max_score:
            max_ind = i
            max_mask = mask
            cropped = croped_image
            max_score = score
            max_bbox = bbox
            best_image = image

    return best_image, path_list[max_ind], max_ind, max_bbox, cropped, max_mask, max_score


def calculate_mask_from_cropped_bbox(bbox_in_cropped_image, mask, image):
    x_min,y_min, x_max,y_max = bbox_in_cropped_image
    initial_mask = np.zeros(image.shape[:2])
    initial_mask[y_min:y_max+1,x_min:x_max+1] = mask
    return initial_mask


def scene_matching(scene_dir: Path, render_dir, result_dir: Path, mesh: o3d.geometry.TriangleMesh):
    render_iter = render_dir.glob("render_*")
    path_list = list(render_iter)
    directions = np.load(render_dir / "directions.npy")
    scene_file_list = list(scene_dir.glob("*.npz"))
    n = 0
    for file in scene_file_list:
        scene_results = np.load(file)
        best_image, render_path, max_ind, max_bbox, cropped, max_mask, score = get_best_render_match(
            path_list, scene_results)
        # save the results
        x_min, y_min, x_max, y_max = max_bbox
        initial_mask = np.zeros(best_image.shape[:2])
        initial_mask[y_min:y_max+1, x_min:x_max+1] = max_mask
        depth_path = render_path.parent /render_path.name.replace("render", "depth")
        direction, rot = directions[max_ind][:-1], directions[max_ind][-1]
        depth, _, _ = get_depth_render(direction, rot, mesh, result_dir, n)
        depth_mask = calculate_mask_from_cropped_bbox(max_bbox, max_mask, best_image)
        np.save(result_dir/(f"depth_mask_{n}.npy"), depth_mask)
        n += 1

