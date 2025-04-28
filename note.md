The task is to find a way to label the objects for grasping. The objects are often represented in the form of 3D meshes. 

We would like to use the objects in simulations and initial meshes obtained from 3D scans can be very complicated. Physical engines like mujoco only 
works with convex hulls of the objects. There is a technique called convex decomposition that represents the body as a set of convex meshes, but it 
produces lots of convex bodies. It is more convenient to use approximate convex decomposition. It will transform the initial mesh into the set of simple meshes. 




The idea is to estimate the object position in the image. 

1. Render the model from different directions. 
2. Get the bounding boxes in renders and the scene. Crop the object from renders and the scene.
3. Resize the cropted parts to match.
4. Mask the area occluded by hand.
5. Use some comparison metric to get the best match.
6. Move the hand area to the model.
7. Optional: think about the way to get more points using the information from the mediapipe, object opposite view or something else, basically do not simply consider the projection.


The code should have the 4 main components:
- the render generator
- scene processing 
- render searcher
- point cloud builder


Render generator:
I set the camera at a position that is far enough from the object and rotate the object to get different angles. The rotation is set by a vector in the object coordinates that I align this the global z axis. I also rotate the object around the global z axis. 
Do I need to save depth images or I can get one after the search.


Scene processing:
The GroundingDINO model performs rather good, at least for hammers. It finds the bounding box to focus the next steps. 
The mediapipe model have several issues. It can find the wrong hand - currently I am trying to force it to find the hand that is closer to the bounding box of the object. 
The SAM model is trying to segment all the objects in the bounding box - therefore it han sometimes segment the wrong object. I can try to segment several objects try to search render for all of them and pick the best of the bests, hoping that it would be the desired object.



TODO:
1. Algorithm for point cloud refinement to make the labels smooth.
2. Increase the usage of mediapipe to get labels for functional areas and handaling areas. Use the depth estimation and try to build mask for opposite direction or use some other symmetry of the object.
3. Consider the hand mask building with something more than bounding box.
4. Consider the translations in render building
5. Try other methods in the render/scene comparison.
 

