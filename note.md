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

