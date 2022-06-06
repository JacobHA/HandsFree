# HandsFree
*3D object manipulation with hands via camera*

(Careful, the code is still very messy... refactoring in progress.)

A simple demo (with SHOW_SELFIE = True):
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

<a href="https://www.buymeacoffee.com/jadamczyk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>


# TODO:
- Much refactoring + cleanup needed
    - [ ] Zoom bug (two hands): zooming switches direction - b/c it cannot tell which hand was detected first.. need to order so consistently L then R
    - [ ] Zoom bug: ensure object stays centered while zooming
    - [x] Two hand bug: two hands at first does not work
    - [ ] Separate the functions for zoom / pan / rotate / etc.
    - [x] Add a smoothing function
    - [x] Fix the MIN_WAITING_FRAMES
    - [x] Track both left and right hand openness separately
- [x] Ask user which camera to use and save it (see next)
- [ ] Create a default settings/preferences file for python app

- [ ] Add keyboard interaction
- [ ] Add voice interaction
- Clean up PyQT applications
    - [ ] Create an executable file
    - [ ] Create a default settings file for QT app
- Interact with CAD (multi-object) files
    - [ ] Allow for "pick and place" of parts
    - [ ] Use exploded view
- [ ] View cross-sections by using hand as slider and plane definer (WIP)

- [x] Create a dependency graph for the project. (At least a requirements.txt file)
