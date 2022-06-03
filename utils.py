import numpy as np

def smooth(y, box_pts):
    # box = np.ones(box_pts)/box_pts
    # y_smooth = np.convolve(y, box, mode='same')
    # return y_smooth
    return np.convolve(y, np.ones(box_pts), 'valid') / box_pts


def cross_prod_mat(v):
    VX = np.zeros((3,3))
    VX[0] = [0, -v[2], v[1]]
    VX[1] = [v[2], 0, -v[0]]
    VX[2] = [-v[1], v[0], 0]
    
    return VX

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))
    
def rot_mat(vecA, vecB):
    vecV = np.cross(vecA, vecB)
    s = vecV.dot(vecV) # sin of rotation angle
    c = vecA.dot(vecB) # cos of rotation angle
    matV = cross_prod_mat(vecV) 
    return np.identity(3) + matV + matV@matV / ( 1 + c )

def rot_mat_AxisAngle(axis, angle):
    assert axis.shape == (3,)
    return np.cos(angle) * np.identity(3) + np.sin(angle) * cross_prod_mat(axis) + (1 - np.cos(angle))*np.outer(axis,axis)

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

from collections import Counter
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