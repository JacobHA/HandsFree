# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 00:20:41 2021

@author: Jacob Adamczyk

To Zoom: Pinch fingers (right hand)
To Rotate: Set up an axis with first finger and thumb (left hand)
To Pan: Raise both hands, and move one (left/right has reversed controls)
"""

print(__doc__)

import cv2
import mediapipe as mp
from vedo import *
from time import sleep
from google.protobuf.json_format import MessageToDict

import numpy as np
from utils import *


y_unit_vec = np.array([0, 1, 0])
z_unit_vec = np.array([0, 0, 1])

STL_name = r'A.stl'

v = Mesh(STL_name)

avg_model_size = v.averageSize()
v.scale(avg_model_size * 0.00001)
dim_scale = np.mean(v.scale())*100
print(avg_model_size)
cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))

# Define a plane that goes through the origin and is parallel to the xy-plane
# plane = plane(pos=(0,0,0), normal=(0,0,1))
# Plot this plane

pl = Plane(v.centerOfMass(), normal=[0,1,0], s=[dim_scale,dim_scale], c='black', alpha=0.3)

plt = show(v, pl, axes=4, viewup='z', camera=cam, interactive=True)

