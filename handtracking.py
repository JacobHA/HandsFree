# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 00:20:41 2021

@author: Jacob Adamczyk

To Zoom: Pinch fingers (right hand)
To Rotate: Set up an axis with first finger and thumb (left hand)
To Pan: Raise both hands, and move one (left/right has reversed controls)
"""

print(__doc__)

import traceback
import cv2
import mediapipe as mp
from vedo import *
from time import sleep

import numpy as np
from utils import *

MEMORY_DEBUG = False

MAX_NUM_HANDS = 2

MAX_TRACKING_TIME = 10
SMOOTHING_INTERVAL = 3
MIN_WAITING_FRAMES = 4

assert SMOOTHING_INTERVAL <= MIN_WAITING_FRAMES

RESET_WAITING_FRAMES = 20
EPSILON_NOISE = 1E-3
FINGER_TOUCHING_RADIUS = 0.07
MOTION_EPSILON = 0.0025
PAUSE_FACTOR = 5

ZOOM_THRESHOLD = 0 #5E-4
ZOOM_EPSILON = 0.01
ZOOM_HARDNESS = 10


ROTATION_SENSITIVITY = 1000
ROTATION_EPSILON = 1e-2
ROTATION_HARDNESS = 4

PANNING_EPSILON = 0.1
PANNING_HARDNESS = 4
PANNING_SENSITIVITY = 2
PANNING_Z_SENSITIVITY = 1.5
ZOOM_SENSITIVITY = 0.1 # effectively how many loop iterations must be done (i.e. ms waited) to acheive zoom factor
INITIAL_RESCALE = 0.00001


SHOW_SELFIE = True#False


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hand_status = MaxSizeList(MIN_WAITING_FRAMES)
open_status = MaxSizeList(RESET_WAITING_FRAMES)

last_N_thumbs = MaxSizeList(MAX_TRACKING_TIME)
last_N_indexes = MaxSizeList(MAX_TRACKING_TIME)
last_N_middles = MaxSizeList(MAX_TRACKING_TIME)
last_N_middle_palms = MaxSizeList(MAX_TRACKING_TIME)
N_paused = MaxSizeList(MAX_TRACKING_TIME * PAUSE_FACTOR)

x_unit_vec = np.array([1, 0, 0])
y_unit_vec = np.array([0, 1, 0])
z_unit_vec = np.array([0, 0, 1])

#TODO: Make a camera clean up function
# TODO: make function for setting up the stl in the plot
STL_name = r'A.stl'

v = Mesh(STL_name)

avg_model_size = v.averageSize()
v.scale(avg_model_size * INITIAL_RESCALE)

cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))


image = None
display_message = "Firing up..."   
status_message = Text2D(display_message, pos="top-center", font=2, c='w', bg='b3', alpha=1)

plt = show(v, status_message, axes=4, viewup='z', camera=cam, interactive=False)

if MEMORY_DEBUG:
    tracemalloc.start()

camera_index = read_index('camera_index.txt')

try:        
    cap = cv2.VideoCapture(camera_index)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=MAX_NUM_HANDS) as hands:
        while cap.isOpened():

            pause_updates = False
            zoom = 1
            display_message = "" 


            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
        
            if SHOW_SELFIE:
                # Correct the coloring and orientation of image 
                image.flags.writeable = False # Performance improvement
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # TODO: is this redundant?
                image.flags.writeable = False
                # Draw the hand annotations on the image.
                multihand_results = results.multi_hand_landmarks

                if multihand_results is None:
                    pass

                else:
                    for hand_landmarks in multihand_results:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the image.
                cv2.imshow('HandsFree Camera', image)
                cv2.waitKey(1)


            del image # this cuts down on memory by a lot; ~1200KB -> ~400KB !

            multihand_results = results.multi_hand_landmarks

            output = data_collector(results, last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms)

            if output is None:
                display_message = 'No hands detected.'

                # i.e. no hands detected
                # CLear out the lists. This saves on memory and more importantly, 
                # does not reference old positions when a hand re-appears
                last_N_thumbs.clear()
                last_N_indexes.clear()
                last_N_middles.clear()
                last_N_middle_palms.clear()
                
                pause_updates = True
                display_message = "Updates paused"

                hand_status.append(None)

            else:

                display_message, hands_present, open_hands, location_data = output
                last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms = location_data
                hand_status.append(hands_present)
                # print(hand_status)
                # print(last_N_indexes)
                # Check last_N_indexes to see if the hand(s) are stationary, meaning that we should pause:
                
                if len(last_N_indexes) > MIN_WAITING_FRAMES:
                    x=is_stationary(last_N_indexes, MOTION_EPSILON)
                    if np.all(x):
                        N_paused.append(True)
                        display_message = 'Pausing...'
                        status_message.text(display_message)
                        plt.show(v, status_message, camera = cam, interactive=False) # important line!           
                        # Cheap way of pausing
                        # time.sleep(1)
                        hand_status += ['Paused']*MAX_TRACKING_TIME
                        last_N_thumbs.clear()
                        last_N_indexes.clear()
                        last_N_middles.clear()
                        last_N_middle_palms.clear()
                        continue
                            
                if hand_status[-MIN_WAITING_FRAMES:] == ['Both']*MIN_WAITING_FRAMES and len(last_N_indexes) >= MIN_WAITING_FRAMES:
                    
                    display_message = "Panning & Zooming"
                    # Extract the left and right hand from the two-hand data:
                    arr = np.array(last_N_indexes)
                    last_N_indexes_L = arr[:, 0, :]
                    last_N_indexes_R = arr[:, 1, :]
                    
                    LR_diff = np.array(last_N_indexes_L) - np.array(last_N_indexes_R)

                    left_change = np.array(last_N_indexes_L[1]) - np.array(last_N_indexes_L[0])
                    right_change = np.array(last_N_indexes_R[1]) - np.array(last_N_indexes_R[0])

                    change = np.mean([left_change, right_change], axis=0)
                    change = np.array([0, change[0], -PANNING_Z_SENSITIVITY * change[1]])
                    # index_change[2] = 0 # set z-axis change to zero            
                    # First check that fingers are not closed: i.e. that we do not want any action
                    change *= sigmoid(change, threshold=PANNING_EPSILON, hardness=PANNING_HARDNESS)
                    v.shift(PANNING_SENSITIVITY * change)
                    # Also use both hands to zoom:
                    xy_change=(LR_diff[1] - LR_diff[0])[0:1].sum()

                    xy_change *= sigmoid(xy_change, threshold = ZOOM_EPSILON, hardness=ZOOM_HARDNESS)
                    zoom_factor = (1 + xy_change) ** (1/ZOOM_SENSITIVITY) # outer plus sign bc pinch out means zoom in
                    zoom = zoom * zoom_factor
                    v.scale(zoom)
                                   
                    plt.show(v, status_message, camera = cam, interactive=False) # important line!           


                    
                elif hand_status[-MIN_WAITING_FRAMES:] == ['Right']*MIN_WAITING_FRAMES:# and len(last_N_indexes) >= MIN_WAITING_FRAMES:
                    arr = np.array(last_N_indexes)
                    print(arr.shape)
                    # Change zoom multiplier based on fingers distance changing (open/close thumb and index)
                    display_message = "Zooming"
                    index_pos = smooth(last_N_indexes, SMOOTHING_INTERVAL)# np.array(last_two_indexes).mean(axis=0)
                    thumb_pos = smooth(last_N_thumbs, SMOOTHING_INTERVAL)#np.array(last_two_thumbs).mean(axis=0)
                  
                    N_indexes = np.array(index_pos)
                    N_thumbs = np.array(thumb_pos)
                    last_two_thumb_index_dists = np.linalg.norm(N_thumbs - N_indexes, axis=1)
                    zoom *= ((1 + (last_two_thumb_index_dists[1] - last_two_thumb_index_dists[0]))) ** (1/ZOOM_SENSITIVITY) # outer plus sign bc pinch out means zoom in
            
                    v.scale(zoom)
                    
                    plt.show(v, status_message, camera = cam, interactive=False) # important line!           

                    

                # TODO: incorporate open status
                elif hand_status[-MIN_WAITING_FRAMES:] == ['Left']*MIN_WAITING_FRAMES and len(last_N_indexes) >= MIN_WAITING_FRAMES:
                    # generate the last smoothed point
                    # print('here')
                    arr = np.array(last_N_indexes)
                    print(arr.shape)
                    index_pos = smooth(last_N_indexes, SMOOTHING_INTERVAL)# np.array(last_two_indexes).mean(axis=0)

                    # Calculate rotation matrix and extract angles

                    display_message = "Rotating"

                    indexes = np.array(last_N_indexes)# - np.array(last_two_indexes_R)

                    change_vector = indexes[1] - indexes[0]
                    normal_to_rotate = change_vector[1] * y_unit_vec + change_vector[0] * z_unit_vec

                    angle_to_rotate = np.linalg.norm(change_vector)
                    # Apply a dampening factor so that small changes don't cause the camera to spin too fast:
                    angle_to_rotate = angle_to_rotate*sigmoid(angle_to_rotate, threshold = ROTATION_EPSILON, hardness=ROTATION_HARDNESS)

                    v.rotate(angle = angle_to_rotate*ROTATION_SENSITIVITY, axis = normal_to_rotate, point=v.centerOfMass())# v.pos())#[::-1]) #bc of axis weirdness
                    
                    plt.show(v, status_message, camera = cam, interactive=False) # important line!           

                elif open_status == [False]*RESET_WAITING_FRAMES:
                    display_message = "Resetting"
                                            
                    v = Mesh(STL_name)

                    avg_model_size = v.averageSize()
                    v.scale(avg_model_size * INITIAL_RESCALE)
                    
                    cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))
                    plt = show(v, status_message, axes=4, viewup='z', camera=cam, interactive=False)
                    

            status_message.text(display_message)
            plt.show(v, status_message, camera = cam, interactive=False) # important line!           

            if MEMORY_DEBUG:
                snapshot = tracemalloc.take_snapshot()
                display_top(snapshot) 

except Exception as e:
    # This enables the camera to be cleaned up if there are any errors
    print('Caught an exception: ' + str(e))
    traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    pass

cap.release()
cv2.destroyAllWindows()

# interactive().close() # Not sure what this does..
