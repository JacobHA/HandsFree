import numpy as np
from google.protobuf.json_format import MessageToDict

def smooth(y, box_pts):
    # box = np.ones(box_pts)/box_pts
    # y_smooth = np.convolve(y, box, mode='same')
    # return y_smooth
    y=np.array(y)
    # print(y)
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

def hand_open(open_tracking_list, wait_time):
    return open_tracking_list == [True]*wait_time

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

        hand_detected = (MessageToDict(mediapipe_results.multi_handedness[0])['classification'][0]['label'])
        multihand_results = mediapipe_results.multi_hand_landmarks
        NUM_HANDS_PRESENT = len(multihand_results)

        # keep track of (which) hands are present in the image frame
        if NUM_HANDS_PRESENT == 1:
            hands_present = hand_detected
            display_message = f"Tracking {hands_present} Hand"

        elif NUM_HANDS_PRESENT == 2:
            hands_present = 'Both'
            display_message = "Tracking Both Hands"

        else:
            raise ValueError('More than 2 hands detected. This is not supported. Check the variable MAX_NUM_HANDS.')

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

 

        # open_status = hand_open(middle_finger_open_list, MIN_WAITING_FRAMES))
        open_status=None
        print(len(last_N_indexes))
        location_data = (last_N_thumbs, last_N_indexes, last_N_middles, last_N_middle_palms)
        return display_message, hands_present, open_status, location_data

    else:       # No hands detected
        return None











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