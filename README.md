# HandsFree
STL manipulation with hands via camera

A simple demo:
![gif of demo](HandsFree-demo.gif)

# Dependencies
- Mayavi
- MediaPipe
- PyQT5
- VTK
- Vedo

## Check out Vedo and MediaPipe:
These codebases power the bulk of this project.
- https://github.com/marcomusy/vedo
-- Vedo is a Python library for visualizing and manipulating data in 3D. The maintainer (Marco Musy) is very helpful and has included a lot of examples to get started for just about any data visualization task.
- https://google.github.io/mediapipe/
-- MediaPipe was created by Google. Its main purpose is to perform pose estimation (hand, face, body, etc.) and process them with a set of computer vision algorithms. One great application of this is to detect hand gestures (there are several cool projects using it to "read" sign language).

# TODO:
- Create a dependency graph for the project. (At least a requirements.txt file)
- Create an executable file
- Clean up PyQT applications
- Ask user which camera
- Create a default settings file for QT app
- Slice the object (for cross sectional views)
- Interact with CAD files