import numpy as np
from google.protobuf.json_format import MessageToDict
from vedo import *


cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))


class ObjectDisplayer:
    def __init__(self, filename, camera=cam, initial_msg='Firing up...', axes=4, init_scale=1):
        self.filename = filename
        self.axes = axes
        self.init_scale = init_scale
        self.camera = camera
        self.is_locked = False
        self.object = Mesh(filename)
        self.status_message = Text2D(initial_msg, pos="top-center", font=2, c='w', bg='b3', alpha=1)

        self.initial_display()

    # Display function
    def show_object(self, new_object=None):
        if not self.is_locked:
            if new_object is None:
                new_object = self.object

            _ = show(new_object, self.status_message, axes=self.axes, viewup='z', camera=self.camera, interactive=False)
        else:
            # There is probably a smarter way to do this so that we do not have to calculate things all the time
            pass

    # Initial functions
    def initial_setup(self):
        # Do fresh import of object
        self.object = Mesh(self.filename)
        model_size = self.object.averageSize()
        self.object.scale(self.init_scale * model_size)
        self.object.shift(-self.object.centerOfMass())
        
    def initial_display(self):
        self.initial_setup()
        # self.initial_shift()
        self.show_object()

    # Update functions
    def update_message(self, message):
        self.status_message.text(message)
        
    def shift_object(self, shift_vector):
        self.object.shift(shift_vector)
        self.show_object()

    def scale_object(self, scale):
        self.object.scale(scale)
        self.show_object()

    def rotate_object(self, angle, axis, point=None):
        if point is None:
            point=self.object.centerOfMass()

        self.object.rotate(angle=angle, axis=axis, point=point)
        self.show_object()

    # Keyboard functions:
    def lock(self):
        self.is_locked = not self.is_locked #True
        msg = 'Locked (press \"l\" again to unlock)' if self.is_locked else 'Unlocked'
        self.update_message(msg)
        _ = show(self.object, self.status_message, axes=self.axes, viewup='z', camera=self.camera, interactive=False)



# def init_show(STL_name, rescale, msg='', camera=cam):
#     status_message = Text2D(msg, pos="top-center", font=2, c='w', bg='b3', alpha=1)

#     v = Mesh(STL_name)
#     avg_model_size = v.averageSize()
#     v.scale(avg_model_size * rescale)

#     # plt.show(v, status_message, camera = cam, interactive=False) 
#     return v   
    
# def show_object(object, msg='', camera=cam):
#     # status_message.text(msg)
#     status_message = Text2D(msg, pos="top-center", font=2, c='w', bg='b3', alpha=1)

#     _ = show(object, status_message, camera = camera, interactive=False)   


def smooth(y, box_pts):

    y=np.array(y)
    return np.array([np.convolve(y_i, np.ones(box_pts), 'valid') / box_pts for y_i in y.T])


class MaxSizeList(list):
    """
    Reduce memory consumption by only monitoring maxlen elements
    """
    def __init__(self, maxlen):
        self._maxlen = maxlen

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeList, self).append(element)


class MaxSizeDict(dict):
    """
    Reduce memory consumption by only monitoring "maxlen" elements
    """
    def __init__(self, maxlen):
        self._maxlen = maxlen

    def append(self, key, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeDict, self).append(key, element)


def filter_noise_below(arr, eps):
    return np.where(arr > eps, arr, 0)

def within_volume_of(points, radius):
    # result = False
    midpoint = np.average(points, axis=0)
    result = [np.linalg.norm(point - midpoint) < radius for point in points]
    return result == [True]*len(points) # needs re written

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def is_touching(distance, threshold):
    return distance < threshold

def hand_open(middle_tip, middle_palm):
    return middle_tip[1] < middle_palm[1]

def sigmoid(x, hardness=1, threshold=0):
    return 1 / (1 + np.exp(-hardness*(x-threshold)))

def read_index(filename):

    with open(filename, 'r') as f:
        
        idx_str = f.read()

        if idx_str == '':
            idx = 0
            print('No index found. Using default value (0).')


        else:
            idx = int(idx_str)
            print(f'Using chosen value ({idx}) for camera index.')


    return idx




THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
MIDDLE_TIP_INDEX = 12 
MIDDLE_PALM_INDEX = 9


def data_collector(mediapipe_results, last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms):
    # First, ensure the results aren't empty, then proceed with data processing/parsing
    if mediapipe_results.multi_hand_landmarks:
        hands_present = None
        display_message = None
        open_status = None

        hand_detected = (MessageToDict(mediapipe_results.multi_handedness[0])['classification'][0]['label'])
        multihand_results = mediapipe_results.multi_hand_landmarks
        NUM_HANDS_PRESENT = len(multihand_results)

        # First we need to check if the number of hands has switched. If so, we must
        # clear out the old list and restart (otherwise we will be concatenating
        # arrays of different shapes)

        arr = np.array(last_N_indexes)
        if len(arr.shape) == 2: # One hand is present (hand-time data, dimensions)
            if NUM_HANDS_PRESENT != 1:
                # Clear out all input arrays before proceeding with data collection:
                for list_name in [last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms]:
                    list_name.clear()

        if len(arr.shape) == 3: # Two hands are present (hand-time data, hand_num, dimensions)
            if NUM_HANDS_PRESENT != 2:
                for list_name in [last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms]:
                    list_name.clear()


        # keep track of (which) hands are present in the image frame
        if NUM_HANDS_PRESENT == 1:
            hands_present = hand_detected
            display_message = f"Tracking {hands_present} Hand"

        elif NUM_HANDS_PRESENT == 2:
            hands_present = 'Both'
            display_message = "Tracking Both Hands"

        else:
            print('More than 2 hands detected. This is not supported. Check the variable MAX_NUM_HANDS.')


        for hand_num, hand_landmarks in enumerate(multihand_results):
            # This will iterate over each hand 
            # Gather finger location data, save on some memory by only iterating thru only what we care about
           
            for landmark_index, finger_positions_list in zip([THUMB_TIP_INDEX, INDEX_TIP_INDEX, MIDDLE_TIP_INDEX, MIDDLE_PALM_INDEX], 
                                                                                    [last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms]):
                if hand_num == 0: # One hand present
                    finger_positions_list.append([
                        hand_landmarks.landmark[landmark_index].x,
                        hand_landmarks.landmark[landmark_index].y,
                        hand_landmarks.landmark[landmark_index].z,
                        ])

                if hand_num == 1: # Both hands present; tack on the other hand's data
                    aux_list = [hand_landmarks.landmark[landmark_index].x, hand_landmarks.landmark[landmark_index].y, hand_landmarks.landmark[landmark_index].z]

                    finger_positions_list[-1] = [finger_positions_list[-1], aux_list]
                    # if hand_detected == 'Left': #left hand was detected second, need to reverse the order of all arrays
                    #     finger_positions_list = finger_positions_list[-1::]

 
        arr = np.array(last_N_indexes)
        if len(arr.shape) == 2: # One hand is present (hand-time data, dimensions)
            open_status = hand_open(last_N_middles[-1], last_N_middle_palms[-1])

        elif len(arr.shape) == 3: # Two hands are present (hand-time data, hand_num, dimensions)
            
            # Extract the left and right hand from the two-hand data:
            _ = np.array(last_N_middles)
            last_N_middles_L = _[-1, 0, :]
            last_N_middles_R = _[-1, 1, :]
            _ = np.array(last_N_middle_palms)
            last_N_middle_palms_L = _[-1, 0, :]
            last_N_middle_palms_R = _[-1, 1, :]
            
            open_status = [hand_open(last_N_mids, last_N_palms) for last_N_mids, last_N_palms in zip([last_N_middles_L, last_N_middles_R],[last_N_middle_palms_L, last_N_middle_palms_R])]

        
        location_data = (last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms)
        return display_message, hands_present, open_status, location_data

    else:       # No hands detected
        return None

def one_hand_is_stationary(last_N_pos, epsilon):
    """ Very simple method to check if there has not been much movement recently. """

    assert len(last_N_pos) >= 2, f"Length of last positions must be >= 2. Length: {len(last_N_pos)}."
    pos_arr3d = np.array(last_N_pos).T # So that dimensions are indexed first
    
    stationary_dims = []
    for dimension in pos_arr3d:
        xi_stationary = (np.abs(np.gradient(dimension).mean()) < epsilon)
        stationary_dims.append(xi_stationary)
 
    return stationary_dims == [True]*3

def is_stationary(last_N_pos, epsilon):
    pos_arr3d = np.array(last_N_pos)
    hands = len(pos_arr3d.shape) - 1
    if hands == 2:
        # Loop through each hand
        last_N_L = pos_arr3d[:, 0, :]
        last_N_R = pos_arr3d[:, 1, :]
                    
        return one_hand_is_stationary(last_N_L, epsilon), one_hand_is_stationary(last_N_R, epsilon)
    elif hands == 1:
        return one_hand_is_stationary(last_N_pos, epsilon)

        








# if multihand_results:

#     hand_present = (MessageToDict(results.multi_handedness[0])['classification'][0]['label'])
#     NUM_HANDS_PRESENT = len(multihand_results)

#     # keep track of (which) hands
#     if NUM_HANDS_PRESENT == 1:
#         hand_status.append(hand_present) 
    
#         # save on memory by only iterating thru only what we care about
#         for hand_landmarks in multihand_results: # [multihand_results[val] for val in landmark_index_nums]:
#             # Draw landmarks 
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
#                 # drawing_styles.get_default_hand_landmark_style(),
#                 # drawing_styles.get_default_hand_connection_style())
                
#             # Gather finger location data
#             for tip_index, finger_positions_list in zip([THUMB_TIP_INDEX, INDEX_TIP_INDEX, MIDDLE_TIP_INDEX], 
#                                                         [last_two_thumbs, last_two_indexes, last_two_middles]):
        
#                 # Need 3D to take cross product...
#                 finger_positions_list.append([
#                     hand_landmarks.landmark[tip_index].x,
#                     hand_landmarks.landmark[tip_index].y,
#                     hand_landmarks.landmark[tip_index].z,
#                     ])

#             # Gather palm location data
#             for tip_index, finger_positions_list in zip([ MIDDLE_TIP_INDEX,          MIDDLE_PALM_INDEX],
#                                                         [ middle_tip_vert_positions, middle_palm_vert_positions]):
        
#                 finger_positions_list.append([
#                     hand_landmarks.landmark[tip_index].y,
#                     ]) # only care about y position for these landmarks

#         middle_finger_open_list.append(middle_tip_vert_positions[-1] < middle_palm_vert_positions[-1])
#         # This will tell us if the hand is open or closed ^

#         # If sufficient data has been collected:
#         display_message = f"Tracking {hand_present} Hand"


#     if NUM_HANDS_PRESENT == 2:
#         hand_status.append('Both') 

#         for hand_landmarks, chirality in zip(multihand_results,[last_two_indexes_L, last_two_indexes_R]): # [multihand_results[val] for val in landmark_index_nums]:
#             # Draw landmarks 
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
#                 # drawing_styles.get_default_hand_landmark_style(),
#                 # drawing_styles.get_default_hand_connection_style())
#             # Gather finger location data
#             tip_index = INDEX_TIP_INDEX
                
                                                                            
#             # Need 3D to pan into/out of page..
#             chirality.append([
#                 hand_landmarks.landmark[tip_index].x,
#                 hand_landmarks.landmark[tip_index].y,
#                 hand_landmarks.landmark[tip_index].z,
#                 ])
#             # Gather palm location data
#             for tip_index, finger_positions_list in zip([ MIDDLE_TIP_INDEX,          MIDDLE_PALM_INDEX],
#                                                         [ middle_tip_vert_positions, middle_palm_vert_positions]):
        
#                 finger_positions_list.append([
#                     hand_landmarks.landmark[tip_index].y,
#                     ]) # only care about y position for these landmarks

#         middle_finger_open_list.append(middle_tip_vert_positions[-1] < middle_palm_vert_positions[-1])
#         # This will tell us if the hand is open or closed ^

#     open_status.append(hand_open(middle_finger_open_list, MIN_WAITING_FRAMES))