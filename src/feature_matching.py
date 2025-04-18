import torch
import kornia as K
import kornia.feature as KF
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


_model = None

def get_model():
    global _model
    if _model is None:
        # Load the LoFTR model (pretrained weights)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = KF.LoFTR(pretrained="outdoor").to(device).eval()
    return _model

def transform(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img = np.array(image.convert("L"))  # Convert to grayscale
    img = torch.from_numpy(img).float() / 255.0  # Normalize to [0,1]
    img = img.unsqueeze(0).unsqueeze(0).to(device)  # Add batch dimensions
    return img

def get_matching_transformation(object_image, handeled_image, vis = False):
    model = get_model()
    # Convert images
    img1_tensor = transform(object_image)
    img2_tensor = transform(handeled_image)
    # Pass images through LoFTR model
    with torch.no_grad():
        input_dict = {"image0": img1_tensor, "image1": img2_tensor}
        correspondences = model(input_dict)

    # Extract matching keypoints
    mkpts0 = correspondences["keypoints0"].cpu().numpy()  # Points in image1
    mkpts1 = correspondences["keypoints1"].cpu().numpy()  # Points in image2
    # Estimate transformation using RANSAC
    M, _ = cv2.estimateAffinePartial2D(mkpts1, mkpts0, method=cv2.RANSAC)
    print("M", M)
    if M is None:
        raise ValueError("Could not estimate transformation matrix.")
    if vis:
            # Visualize matches
        img1 = np.array(object_image)
        img2 = np.array(handeled_image.convert("RGB"))

        # Determine resize scale based on height
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_height = min(h1, h2)
        scale1 = target_height / h1
        scale2 = target_height / h2

        # Resize images
        img1_resized = cv2.resize(img1, (int(w1 * scale1), target_height))
        img2_resized = cv2.resize(img2, (int(w2 * scale2), target_height))
        print(img1_resized.shape, img2_resized.shape)

        # Resize keypoints accordingly
        mkpts0_scaled = mkpts0 * scale1
        mkpts1_scaled = mkpts1 * scale2

        # Concatenate images side by side
        concat_img = np.hstack((img1_resized, img2_resized))
        offset = img1_resized.shape[1]
        max_matches = 20
        # Limit number of matches
        num_matches = min(max_matches, len(mkpts0))

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.imshow(concat_img)
        for i in range(num_matches):
            pt1 = mkpts0_scaled[i]
            pt2 = mkpts1_scaled[i].copy()
            pt2[0] += offset  # shift for second image

            color = np.random.rand(3,)
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=1)
            plt.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, s=10)

        plt.axis("off")
        plt.title(f"LoFTR Matches: {num_matches} shown")
        plt.show()
    return M


def bbox_to_corners(bbox):
    x_min, y_min, x_max, y_max = bbox
    corners = np.array([
        [x_min, y_min],  # Top-left
        [x_max, y_min],  # Top-right
        [x_max, y_max],  # Bottom-right
        [x_min, y_max]   # Bottom-left
    ])
    return corners  # Shape: (4, 2)


def apply_homography(H, point):
    """
    Applies homography matrix H to a 2D point (x, y).
    Returns the transformed point in image1 coordinates.
    """
    point_hom = np.array([point[0], point[1], 1.0])
    transformed = H @ point_hom
    transformed /= transformed[2]  # Normalize by last coordinate
    return transformed[:2]

def apply_affine_transform(matrix, points):
    """
    Apply the affine transform to a list of points.
    
    Parameters:
    - matrix: Affine transformation matrix (2x3).
    - points: List or array of points to be transformed (N x 2).
    
    Returns:
    - transformed_points: Transformed points (N x 2).
    """
    points = np.array(points, dtype=np.float32)
    transformed_points = cv2.transform(points[None, :, :], matrix)
    return transformed_points[0]

def transform_corners(corners, H):
    # Convert to homogeneous coordinates (add 1 as last column)
    transformed_corners = []
    for corner in corners:
        transformed_corners.append(apply_homography(H, corner))
    # homogeneous_corners = np.column_stack([corners, np.ones(len(corners))])
    
    # # Apply transformation: M @ [x, y, 1]^T
    # transformed_corners = (M @ homogeneous_corners.T).T  # Shape: (4, 2)
    return transformed_corners

def transform_corners_affine(corners, M):
    # Convert to homogeneous coordinates (add 1 as last column)
    transformed_corners = apply_affine_transform(M, corners)
    # for corner in corners:
    #     transformed_corners.append(apply_homography(H, corner))
    # homogeneous_corners = np.column_stack([corners, np.ones(len(corners))])
    
    # # Apply transformation: M @ [x, y, 1]^T
    # transformed_corners = (M @ homogeneous_corners.T).T  # Shape: (4, 2)
    return transformed_corners
