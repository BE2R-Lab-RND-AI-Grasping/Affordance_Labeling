# Semantic_Object_Segmentation_for_Affordance_Labeling

The tool for labeling of the 3D models using the images of the object usage. 

## Environment configuration
Use `environment.yml` to install dependencies
```bash
conda env create -f environment.yml
conda activate labeling_env
``` 

## How it works

The input for the algorithm is the 3D model of an object and a folder containing the images of the object usage. The images should be of similar objects, but not necessary identical.

The inputs should be organised as follows:
```bash
    |-- data
        |-- model
            |-- object_model.obj
            |-- scenes
                |-- scene_image_0.jpg
                |-- scene_image_1.jpg
                |-- ...
``` 

The pipeline consists of four steps.
### Render generation
The model is used to generate renders from different view points and camera angles. The results are saved into the `renders` folder. The results include:
- `camera_params.json` for storing intrinsic and extrinsic camera parameters from Open3D
- `directions.npy` for stroing the direction and rotation of the camera
- `filelist.json` for storing the list of paths to the render files
- renders from different directions

<img src="./assets/1.png" width="45%"/> <img src="./assets/2.png" width="45%"/>
<img src="./assets/3.png" width="45%"/> <img src="./assets/4.png" width="45%"/>

### Scene processing
Scenes are the images of the object usage. For each scene we use several pretrained networks to find the area of the hand-object interaction. As a first step we use GroundingDINO with object name prompt to find the bounding box of the object in the scene. Then we use the MediaPipe for hand detection and build the hand bounding box. Finally the SAM model is applied to get the segmentation mask of the object, using the bounding box created by GroundingDINO. 

GroundingDINO example:





