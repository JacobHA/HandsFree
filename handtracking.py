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

MEMORY_DEBUG = False

THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
MIDDLE_TIP_INDEX = 12 
MIDDLE_PALM_INDEX = 9

MAX_TRACKING_TIME = 50
SMOOTHING_INTERVAL = 10
MIN_WAITING_FRAMES = 2
RESET_WAITING_FRAMES = 20
EPSILON_NOISE = 1E-3
FINGER_TOUCHING_RADIUS = 0.07
ZOOM_THRESHOLD = 0 #5E-4
ROTATION_SENSITIVITY = 1000
ROTATION_EPSILON = 5e-3
ROTATION_HARDNESS = 4
PANNING_SENSITIVITY = 2
PANNING_Z_SENSITIVITY = 1.5
ZOOM_SENSITIVITY = 0.1 # effectively how many loop iterations must be done (i.e. ms waited) to acheive zoom factor
INITIAL_RESCALE = 0.00001


SHOW_SELFIE = True#False

dimensions = 3

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# drawing_styles = mp.solutions.drawing_styles

thumb_positions = MaxSizeList(MAX_TRACKING_TIME) 
index_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_tip_vert_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_palm_vert_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_finger_open_list = MaxSizeList(MIN_WAITING_FRAMES)
hand_status = MaxSizeList(MIN_WAITING_FRAMES)
open_status = MaxSizeList(RESET_WAITING_FRAMES)

last_two_positions = MaxSizeList(MIN_WAITING_FRAMES * 3) # NUM_FINGERS_NEEDED * NUM_DIMENSIONS
last_two_thumbs = MaxSizeList(MIN_WAITING_FRAMES)
last_two_indexes = MaxSizeList(MIN_WAITING_FRAMES)
# When two hands must be tracked
last_two_indexes_L = MaxSizeList(MIN_WAITING_FRAMES)
last_two_indexes_R = MaxSizeList(MIN_WAITING_FRAMES)

last_two_middles = MaxSizeList(MIN_WAITING_FRAMES)

last_two_thumb_index_vecs = MaxSizeList(MIN_WAITING_FRAMES)
last_two_thumb_index_dists = MaxSizeList(MIN_WAITING_FRAMES)

y_unit_vec = np.array([0, 1, 0])
z_unit_vec = np.array([0, 0, 1])

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

try:        
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9, max_num_hands=2) as hands:
        while cap.isOpened():

            pause_updates = False
            new_zoom, old_zoom = 1,1

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            results = hands.process(image)
        
            # Draw the hand annotations on the image.
            image.flags.writeable = False # Performance improvement
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image.flags.writeable = False
            display_message = "" # So as not to overcrowd with box behind

            multihand_results = results.multi_hand_landmarks


            if multihand_results:

                hand_present = (MessageToDict(results.multi_handedness[0])['classification'][0]['label'])
                NUM_HANDS_PRESENT = len(multihand_results)

                # keep track of (which) hands
                if NUM_HANDS_PRESENT == 1:
                    hand_status.append(hand_present) 
              
                    # save on memory by only iterating thru only what we care about
                    for hand_landmarks in multihand_results: # [multihand_results[val] for val in landmark_index_nums]:
                        # Draw landmarks 
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
                            # drawing_styles.get_default_hand_landmark_style(),
                            # drawing_styles.get_default_hand_connection_style())
                            
                        # Gather finger location data
                        for tip_index, finger_positions_list in zip([THUMB_TIP_INDEX, INDEX_TIP_INDEX, MIDDLE_TIP_INDEX], 
                                                                    [last_two_thumbs, last_two_indexes, last_two_middles]):
                    
                            # Need 3D to take cross product...
                            finger_positions_list.append([
                                hand_landmarks.landmark[tip_index].x,
                                hand_landmarks.landmark[tip_index].y,
                                hand_landmarks.landmark[tip_index].z,
                                ])

                        # Gather palm location data
                        for tip_index, finger_positions_list in zip([ MIDDLE_TIP_INDEX,          MIDDLE_PALM_INDEX],
                                                                    [ middle_tip_vert_positions, middle_palm_vert_positions]):
                    
                            finger_positions_list.append([
                                hand_landmarks.landmark[tip_index].y,
                                ]) # only care about y position for these landmarks

                    middle_finger_open_list.append(middle_tip_vert_positions[-1] < middle_palm_vert_positions[-1])
                    # This will tell us if the hand is open or closed ^

                    # If sufficient data has been collected:
                    display_message = f"Tracking {hand_present} Hand"


                if NUM_HANDS_PRESENT == 2:
                    hand_status.append('Both') 

                    for hand_landmarks, chirality in zip(multihand_results,[last_two_indexes_L, last_two_indexes_R]): # [multihand_results[val] for val in landmark_index_nums]:
                        # Draw landmarks 
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
                            # drawing_styles.get_default_hand_landmark_style(),
                            # drawing_styles.get_default_hand_connection_style())
                        # Gather finger location data
                        tip_index = INDEX_TIP_INDEX
                         
                                                                                        
                        # Need 3D to pan into/out of page..
                        chirality.append([
                            hand_landmarks.landmark[tip_index].x,
                            hand_landmarks.landmark[tip_index].y,
                            hand_landmarks.landmark[tip_index].z,
                            ])
                        # Gather palm location data
                        for tip_index, finger_positions_list in zip([ MIDDLE_TIP_INDEX,          MIDDLE_PALM_INDEX],
                                                                    [ middle_tip_vert_positions, middle_palm_vert_positions]):
                    
                            finger_positions_list.append([
                                hand_landmarks.landmark[tip_index].y,
                                ]) # only care about y position for these landmarks

                    middle_finger_open_list.append(middle_tip_vert_positions[-1] < middle_palm_vert_positions[-1])
                    # This will tell us if the hand is open or closed ^

                open_status.append(hand_open(middle_finger_open_list, MIN_WAITING_FRAMES))

                if len(last_two_thumbs) >= MIN_WAITING_FRAMES:
                    display_message = f"Tracking Both Hands"

                    # generate/grab the last two smoothed points
                    # for finger_positions_list, last_two in zip([thumb_positions, index_positions, middle_positions],
                    #                                             [last_two_thumbs, last_two_indexes, last_two_middles]):
                    #     for dim in range(dimensions): # x,y,z
                    #         last_two.append(
                    #             smooth(np.array(finger_positions_list).T.tolist()[dim], SMOOTHING_INTERVAL)[-2:])

                    # Rather than smoothing, let's just average the last 2 points
                    index_pos = np.array(last_two_indexes).mean(axis=0)
                    thumb_pos = np.array(last_two_thumbs).mean(axis=0)
                    middle_pos = np.array(last_two_middles).mean(axis=0)

                   
                    # find the vector between two points
                    thumb_to_index = index_pos - thumb_pos
                    thumb_to_middle = middle_pos - thumb_pos
                    
                    # Optionally do masking here... it helps prevent the distance from being changed by z coord
                    # thumb_to_index[-1] = 0 # set z coord to zero
                    thumb_to_index *= -1 # offset coord axis weirdness (y goes down)
                    last_two_thumb_index_vecs = np.array(last_two_indexes) - np.array(last_two_thumbs)
                    last_two_thumb_index_dists = np.linalg.norm(last_two_thumb_index_vecs, axis=1)
                    # ^^ Instead of this just do live-time averaging... result += thumb_to_index
                    # result /= 2
            
                    if hand_status == ['Both']*MIN_WAITING_FRAMES and open_status[-1] and len(last_two_indexes_L) > 1:
                        
                        display_message = "Panning"
                        # Pan camera
                        
                        # TODO: change this difference logic:
                        indexes = np.mean([np.array(last_two_indexes_L), np.array(last_two_indexes_R)], axis=0)
                        change = indexes[1] - indexes[0]
                        change = np.array([0, change[0], -PANNING_Z_SENSITIVITY * change[1]])
                        # index_change[2] = 0 # set z-axis change to zero            
                        # First check that fingers are not closed: i.e. that we do not want any action
                        v.shift(PANNING_SENSITIVITY * change)
                       
                    if hand_status == ['Right']*MIN_WAITING_FRAMES and open_status[-1] and len(last_two_thumb_index_dists) > 1:

                        # Change zoom multiplier based on fingers distance changing (open/close thumb and index)
                        display_message = "Zooming"
        
                        new_zoom *= ((1 + (last_two_thumb_index_dists[1] - last_two_thumb_index_dists[0]))) ** (1/ZOOM_SENSITIVITY) # outer plus sign bc pinch out means zoom in
               
                        v.scale(new_zoom)

                    if hand_status == ['Left']*MIN_WAITING_FRAMES and open_status[-1] and len(last_two_indexes) > 1:

                        # Calculate rotation matrix and extract angles

                        display_message = "Rotating"

                        indexes = np.array(last_two_indexes)# - np.array(last_two_indexes_R)

                        change_vector = indexes[1] - indexes[0]
                        normal_to_rotate = change_vector[1] * y_unit_vec + change_vector[0] * z_unit_vec
                        # normal_to_rotate = np.cross(thumb_to_middle, thumb_to_index)
                        # np.cross(np.average(last_two_thumb_index_vecs,axis=0), z_unit_vec) # always crossing it into the screen..check sign later
                        # angle_to_rotate = (last_two_thumb_index_dists).mean()
                        angle_to_rotate = np.linalg.norm(change_vector)
                        # Apply a dampening factor so that small changes don't cause the camera to spin too fast:
                        angle_to_rotate = angle_to_rotate*sigmoid(angle_to_rotate, threshold = ROTATION_EPSILON, hardness=ROTATION_HARDNESS)

                        v.rotate(angle = angle_to_rotate*ROTATION_SENSITIVITY, axis = normal_to_rotate, point=v.centerOfMass())# v.pos())#[::-1]) #bc of axis weirdness

                    if open_status == [False]*RESET_WAITING_FRAMES:
                        display_message = "Resetting"
                                                
                        v = Mesh(STL_name)

                        avg_model_size = v.averageSize()
                        v.scale(avg_model_size * INITIAL_RESCALE)
                        
                        cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))
                        plt = show(v, status_message, axes=4, viewup='z', camera=cam, interactive=False)
                        

        

            else: # i.e. no hands detected
                # CLear out the lists. This saves on memory and more importantly, 
                # does not reference old positions when a hand re-appears
                last_two_indexes_L.clear()
                last_two_indexes_R.clear()
                thumb_positions.clear() 
                index_positions.clear()
                last_two_indexes.clear()
                new_zoom = 1
                pause_updates = True
            
            # Show vtk file and camera's image
            if pause_updates:
                display_message = "Updates paused"

            if SHOW_SELFIE:
                cv2.imshow('MediaPipe', image)
                cv2.waitKey(1)
            del image # this cuts down on memory by a lot; ~1200KB -> ~200KB !

            status_message.text(display_message)
            plt.show(v, status_message, camera = cam, interactive=False) # important line!           

            if MEMORY_DEBUG:
                snapshot = tracemalloc.take_snapshot()
                display_top(snapshot) 

except Exception as e:
    # This enables the camera to be cleaned up if there are any errors
    print('Caught an exception: ' + str(e))
    cap.release()
    cv2.destroyAllWindows()
    pass

cap.release()
cv2.destroyAllWindows()

# interactive().close() # Not sure what this does..
