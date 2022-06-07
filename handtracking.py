# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 00:20:41 2021

@author: Jacob Adamczyk

To Zoom: Pinch fingers (right hand)
To Rotate: Set up an axis with first finger and thumb (left hand)
To Pan: Raise both hands, and move one (left/right has reversed controls)

Key Commands:
    'q' - Quit the program
    'r' - Reset to the initial viewing display
    's' - Toggle selfie mode (Work in Progress)
    'l' - Lock/Unlock the object
"""

print(__doc__)

import traceback
import tracemalloc
import cv2
import mediapipe as mp

import numpy as np
from utilities.data_utils import MaxSizeList, data_collector, is_stationary, sigmoid
from utilities.vis_utils import DisplayScene, ManipulableObject
from utilities.settings_utils import read_index
from utilities.memory_utils import display_top

MEMORY_DEBUG = False

MAX_NUM_HANDS = 2

MAX_TRACKING_TIME = 6
SMOOTHING_INTERVAL = 1
MIN_WAITING_FRAMES = 5
PAUSE_FRAMES = 7

assert SMOOTHING_INTERVAL <= MIN_WAITING_FRAMES

RESET_WAITING_FRAMES = 5
EPSILON_NOISE = 1E-3
FINGER_TOUCHING_RADIUS = 0.07
MOTION_EPSILON = 0.003

ZOOM_THRESHOLD = 0 #5E-4
ZOOM_EPSILON = 0.01
ZOOM_HARDNESS = 10


ROTATION_SENSITIVITY = 1000
ROTATION_EPSILON = 5e-3
ROTATION_HARDNESS = 3

PANNING_EPSILON = 0.1
PANNING_HARDNESS = 4
PANNING_SENSITIVITY = 2
PANNING_Z_SENSITIVITY = 1.5
ZOOM_SENSITIVITY = 0.1 # effectively how many loop iterations must be done (i.e. ms waited) to acheive zoom factor
INITIAL_RESCALE = 0.00001


SHOW_SELFIE = True#False


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hand_status = MaxSizeList(MAX_TRACKING_TIME)

last_N_thumbs = MaxSizeList(MAX_TRACKING_TIME)
last_N_indexes = MaxSizeList(MAX_TRACKING_TIME)
last_N_middles = MaxSizeList(MAX_TRACKING_TIME)
last_N_middle_palms = MaxSizeList(MAX_TRACKING_TIME)
paused = MaxSizeList(PAUSE_FRAMES)

x_unit_vec = np.array([1, 0, 0])
y_unit_vec = np.array([0, 1, 0])
z_unit_vec = np.array([0, 0, 1])

#TODO: Make a camera clean up function

# """Create a Volume from a numpy.mgrid"""
# import numpy as np
# from vedo import Volume, Text2D, show


# import scipy.io as spio
# import vtk

# phall = spio.loadmat(r'C:\Users\jacob\OneDrive\Desktop\Cancer Biophysics\phall.mat')['PHALLOIDIN_IMG']
# dapi = spio.loadmat(r'C:\Users\jacob\OneDrive\Desktop\Cancer Biophysics\dapi.mat')['DAPI_IMG']

# phall[phall <= 4] = 0
# # phall_max = np.where(phall <= 20)[0].max()
# # phall[phall >= 20] = phall_max

# dapi[dapi <= 15] = 0
# # dapi_max = np.where(phall <= 200)[0].max()
# # dapi[dapi >= 200] = dapi_max


# import vtk.util.numpy_support as numpy_support


# # Stretch out z axis:

# X, Y, Z = np.mgrid[:512, :512, :10]
# # Distance from the center at (15, 15, 15)
# # scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2) /225
# scalar_field1 = phall[X,Y,Z]
# scalar_field2 = dapi[X,Y,Z]

# vol1 = Volume(scalar_field1, mapper='smart',spacing=(2.5,0.176,0.176))#, mapper='smart')#,mode=1)
# vol2 = Volume(scalar_field2, mapper='smart',spacing=(2.5,0.176,0.176))#, mapper='smart')#,mode=1)

# # Combine the two volume objects
# # vol = vol1.boolean("+", vol2).flat()#.addScalarBar()

# print(vol1.bounds())
# print(vol1)


STL_name = r'A.stl'

# Create the displayer:
# obj = ObjectDisplayer([vol1,vol2], init_scale=INITIAL_RESCALE)
# Create the scene for objects to live in:
scene = DisplayScene()

# Add an object, the stl file:
stl_object = ManipulableObject(filename=STL_name, initial_scale=INITIAL_RESCALE)
hollow_cube = ManipulableObject(filename=r'hollow_cube.stl', initial_scale=INITIAL_RESCALE)
# scene.add_object(stl_object)
scene.add_object(hollow_cube)
scene.display_objects()


image = None

if MEMORY_DEBUG:
    tracemalloc.start()

camera_index = read_index('camera_index.txt')



# Begin main loop of camera read-in, data processing, and object manipulation:
try:        
    cap = cv2.VideoCapture(camera_index)

    with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85, max_num_hands=MAX_NUM_HANDS) as hands:
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

            key_press = cv2.waitKey(1)
        
            del image # this cuts down on memory by a lot; ~1200KB -> ~400KB !

            multihand_results = results.multi_hand_landmarks

            output = data_collector(results, last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms)

            if key_press == ord('l'):
                # Lock/unlock all the objects
                for object in scene.objects:
                    object.swap_lock()
                scene.display_objects()
                # TODO: clear all lists

            if key_press == ord('r'):
                # Reset the object
                stl_object.initial_setup()
                hollow_cube.initial_setup()
                scene.display_objects()

            if key_press == ord('q'):
                # Quit
                exit()

            # if key_press == ord('s'):
            #     # Toggle selfie mode
            #     # TODO: some work needs done here to turn camera back on
            #     # Close the camera window:
            #     cv2.destroyWindow('HandsFree Camera')

            # if key_press == ord('p'):
            #     # Precision mode

                

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
                scene.update_message(display_message)
                scene.display_objects()
                hand_status.append(None)

            else:

                display_message, hands_present, open_hands, location_data = output
                last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms = location_data
                hand_status.append(hands_present)

                # Check last_N_indexes to see if the hand(s) are stationary, meaning that we should pause (or reset if hands are closed):
                
                if len(last_N_indexes) == MAX_TRACKING_TIME: # This line can change the effective pause check time (how long should hands be stationary/closed)
                    if np.all(is_stationary(last_N_indexes, MOTION_EPSILON)):
                        both_hands_closed = not np.all(open_hands)
                        if both_hands_closed:
                            
                            display_message = 'Resetting...'
                            scene.update_message(display_message)
                            scene.initial_display() 
                            
                        else:
                            display_message = 'Pausing...'
                            scene.update_message(display_message)
                            scene.display_objects()

                        hand_status += ['Paused']*MAX_TRACKING_TIME
                        paused.append(True)
                        last_N_thumbs.clear()
                        last_N_indexes.clear()
                        last_N_middles.clear()
                        last_N_middle_palms.clear()
                        continue
                    else:
                        paused.append(False)
                        

                    if not np.all(paused) and not np.all([obj.is_locked for obj in scene.objects]) and len(hand_status) >= MIN_WAITING_FRAMES:
                                
                        if hand_status[-MIN_WAITING_FRAMES:] == ['Both']*MIN_WAITING_FRAMES:
                            
                            display_message = "Panning & Zooming"
                            scene.update_message(display_message)

                            # Extract the left and right hand from the two-hand data:
                            arr = np.array(last_N_indexes)
                            last_N_indexes_L = arr[:, 0, :]
                            last_N_indexes_R = arr[:, 1, :]
                            
                            LR_diff = np.abs(np.array(last_N_indexes_L) - np.array(last_N_indexes_R))

                            left_change = np.array(last_N_indexes_L[1]) - np.array(last_N_indexes_L[0])
                            right_change = np.array(last_N_indexes_R[1]) - np.array(last_N_indexes_R[0])

                            change = np.mean([left_change, right_change], axis=0)
                            change = np.array([0, change[0], -PANNING_Z_SENSITIVITY * change[1]])
                            # index_change[2] = 0 # set z-axis change to zero            
                            # First check that fingers are not closed: i.e. that we do not want any action
                            change *= sigmoid(change, threshold=PANNING_EPSILON, hardness=PANNING_HARDNESS)
                            
                            for object in scene.objects:
                                object.shift_object(PANNING_SENSITIVITY * change)

                            # Also use both hands to zoom:
                            xy_change=(LR_diff[1] - LR_diff[0])[0:1].sum()

                            xy_change *= sigmoid(xy_change, threshold = ZOOM_EPSILON, hardness=ZOOM_HARDNESS)
                            zoom_factor = (1 + xy_change) ** (1/ZOOM_SENSITIVITY) # outer plus sign bc pinch out means zoom in
                            zoom = zoom * zoom_factor

                            for object in scene.objects:
                                object.scale_object(zoom)
                            # scene.assembly.scale_object(zoom)
                                        

                        elif hand_status[-MIN_WAITING_FRAMES:] == ['Left']*MIN_WAITING_FRAMES:# and len(last_N_indexes) >= MIN_WAITING_FRAMES:
                            # generate the last smoothed point
                            arr = np.array(last_N_indexes)
                            # index_pos = smooth(last_N_indexes, SMOOTHING_INTERVAL)*0.1# np.array(last_two_indexes).mean(axis=0)

                            display_message = "Rotating"
                            scene.update_message(display_message)

                            indexes = np.array(last_N_indexes)# - np.array(last_two_indexes_R)

                            change_vector = indexes[1] - indexes[0]
                            # normal_to_rotate = change_vector[0] * y_unit_vec + change_vector[1] * z_unit_vec
                            normal_to_rotate = change_vector[1] * y_unit_vec + change_vector[0] * z_unit_vec

                            angle_to_rotate = np.linalg.norm(change_vector)
                            # Apply a damping factor so that small changes don't cause the camera to spin too fast:
                            angle_to_rotate = angle_to_rotate*sigmoid(angle_to_rotate, threshold = ROTATION_EPSILON, hardness=ROTATION_HARDNESS)

                            for object in scene.objects:
                                object.rotate_object(angle = angle_to_rotate*ROTATION_SENSITIVITY, axis = normal_to_rotate)# v.pos())#[::-1]) #bc of axis weirdness
                            # scene.assembly.rotate_object(angle = angle_to_rotate*ROTATION_SENSITIVITY, axis = normal_to_rotate)# v.pos())#[::-1]) #bc of axis weirdness
                        
                        elif hand_status[-MIN_WAITING_FRAMES:] == ['Right']*MIN_WAITING_FRAMES and len(last_N_indexes) >= MIN_WAITING_FRAMES:
                            arr = np.array(last_N_indexes)[-2] - np.array(last_N_indexes)[-1]
                            arr = -np.array([0, arr[0], -arr[1]]) #location of index finger


                            # scene.show_cross_section()

                        #     scene.show_selector(0.5 * arr)
                        #     # print(np.linalg.norm(arr))
                        #     if np.linalg.norm(arr) < 0.005: # Thumb is touching index
                        #         scene.select_object() # change to mean bw thumb and index
                        #     else:
                        #         for obj in scene.objects:
                        #             obj.unlock()

                        # scene.remove_selector()
                        scene.display_objects()

                            
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
