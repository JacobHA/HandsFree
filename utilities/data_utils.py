from google.protobuf.json_format import MessageToDict
import numpy as np


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
    import numpy as np
    pos_arr3d = np.array(last_N_pos)
    hands = len(pos_arr3d.shape) - 1
    if hands == 2:
        # Loop through each hand
        last_N_L = pos_arr3d[:, 0, :]
        last_N_R = pos_arr3d[:, 1, :]
                    
        return one_hand_is_stationary(last_N_L, epsilon), one_hand_is_stationary(last_N_R, epsilon)
    elif hands == 1:
        return one_hand_is_stationary(last_N_pos, epsilon)

    import numpy as np


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

def hand_open(middle_tip, middle_palm):
    return middle_tip[1] < middle_palm[1]

def sigmoid(x, hardness=1, threshold=0):
    return 1 / (1 + np.exp(-hardness*(x-threshold)))
