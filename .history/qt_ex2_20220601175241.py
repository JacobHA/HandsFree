
"""
A sort of minimal example of how to embed a rendering window
into a qt application.
"""
print(__doc__)

import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import cv2
from vedo import Plotter, Mesh
import numpy as np
from utils import *
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from PyQt5.QtCore import QTimer



THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
MIDDLE_TIP_INDEX = 12 
MIDDLE_PALM_INDEX = 9
MAX_TRACKING_TIME = 50
SMOOTHING_INTERVAL = 10
MIN_WAITING_FRAMES = 10
EPSILON_NOISE = 1E-3
FINGER_TOUCHING_RADIUS = 0.07
ZOOM_THRESHOLD = 0 #5E-4
ROTATION_SENSITIVITY = 10
PANNING_SENSITIVITY = 2
PANNING_Z_SENSITIVITY = 1.5
ZOOM_SENSITIVITY = 0.1 # effectively how many loop iterations must be done (i.e. ms waited) to acheive zoom factor
INITIAL_RESCALE = 0.00001

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# drawing_styles = mp.solutions.drawing_styles

z_unit_vec = np.array([0,0,1])


thumb_positions = MaxSizeList(MAX_TRACKING_TIME) 
index_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_tip_vert_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_palm_vert_positions = MaxSizeList(MAX_TRACKING_TIME)
middle_finger_open_list = MaxSizeList(MIN_WAITING_FRAMES)
hand_status = MaxSizeList(MIN_WAITING_FRAMES)

last_two_positions = MaxSizeList(2 * 3) # NUM_FINGERS_NEEDED * NUM_DIMENSIONS
last_two_thumb_index_vecs = MaxSizeList(2)
last_two_thumb_index_dists = MaxSizeList(2)

def camera_loop(cap, hands, hand_status, thumb_positions,index_positions,middle_tip_vert_positions, middle_palm_vert_positions, middle_finger_open_list, last_two_positions):

    pause_updates = False
    new_zoom, old_zoom = 1,1

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        raise SystemError
        return None

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
        # hand_present = results.multi_handedness[0].classification[0].label  # label 

        hand_present = (MessageToDict(results.multi_handedness[0])['classification'][0]['label'])
        NUM_HANDS_PRESENT = len(multihand_results)
        if NUM_HANDS_PRESENT == 1:
            hand_status.append(hand_present) # keep track of hands
        if NUM_HANDS_PRESENT == 2:
            hand_status.append('Both')

        
        # landmark_index_nums = [THUMB_TIP_INDEX, INDEX_TIP_INDEX, MIDDLE_TIP_INDEX, MIDDLE_PALM_INDEX] 
        # save on memory by only iterating thru what we care about
        for hand_landmarks in multihand_results: # [multihand_results[val] for val in landmark_index_nums]:
            # Draw landmarks 
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,)
                # drawing_styles.get_default_hand_landmark_style(),
                # drawing_styles.get_default_hand_connection_style())
                
            # Gather finger location data
            for tip_index, finger_positions_list in zip([THUMB_TIP_INDEX, INDEX_TIP_INDEX], 
                                                        [thumb_positions, index_positions]):
        
                finger_positions_list.append([
                    hand_landmarks.landmark[tip_index].x,
                    hand_landmarks.landmark[tip_index].y,
                    hand_landmarks.landmark[tip_index].z,
                    ])

            for tip_index, finger_positions_list in zip([ MIDDLE_TIP_INDEX,          MIDDLE_PALM_INDEX],
                                                        [ middle_tip_vert_positions, middle_palm_vert_positions]):
        
                finger_positions_list.append([
                    hand_landmarks.landmark[tip_index].y,
                    ]) # only care about y position for these landmarks
        middle_finger_open_list.append(middle_tip_vert_positions[-1] < middle_palm_vert_positions[-1])
        # This will tell us if the hand is open or closed ^

        # If sufficient data has been collected:
        display_message = "Tracking hand"

        if len(thumb_positions) > MIN_WAITING_FRAMES:
            
            # generate/grab the last two smoothed points
            for finger_positions_list in [thumb_positions, index_positions]:
                for dim in range(3): # x,y,z
                    last_two_positions.append(
                        smooth(np.array(finger_positions_list).T.tolist()[dim], SMOOTHING_INTERVAL)[-2:])

                    
            # Create AOR in xy plane based of thumb-index line and rotate based on distance bw fingers
            last_two_thumb_index_vecs = MaxSizeList(2)
            last_two_thumb_index_dists = MaxSizeList(2)

            for timestep in np.array(last_two_positions).T.reshape(2,2,3):
                thumb_pos, index_pos = timestep # extract the 3d points of thumb, index, middle                   
                # First check that fingers are not closed: i.e. that we do not want any action
                # fingers_touching = within_volume_of([pointA, pointB, pointC], FINGER_TOUCHING_RADIUS)
                # if NUM_HANDS_PRESENT == 2:
                #     pause_updates = True
                # first find the vector between two points
                thumb_to_index = index_pos - thumb_pos
                # Optionally do masking here... it helps prevent the distance from being changed by z coord
                # thumb_to_index[-1] = 0 # set z coord to zero
                thumb_to_index *= -1 # offset coord axis weirdness (y goes down)
                last_two_thumb_index_vecs.append(thumb_to_index)
                last_two_thumb_index_dists.append(np.linalg.norm(thumb_to_index))
                # ^^ Instead of this just do live-time averaging... result += thumb_to_index
                # result /= 2
                

            if hand_status == ['Both']*MIN_WAITING_FRAMES and \
                hand_open(middle_finger_open_list, MIN_WAITING_FRAMES):
                display_message = "Panning"
                # Pan camera
                # Problem is I only have data for one hand seemingly....
                index_posits = []
                for timestep in np.array(last_two_positions).T.reshape(2,2,3):
                    thumb_pos, index_pos = timestep # extract the 3d points of thumb, index, middle       
                    index_posits.append(index_pos)
                index_change = index_posits[1] - index_posits[0]
                index_change = [index_change[2], index_change[0], -PANNING_Z_SENSITIVITY * index_change[1]]

                return display_message, PANNING_SENSITIVITY * np.array(index_change)
                
            if hand_status == ['Right']*MIN_WAITING_FRAMES and \
                hand_open(middle_finger_open_list, MIN_WAITING_FRAMES):

                # Change zoom multiplier based on fingers distance changing (open/close thumb and index)
                display_message = "Zooming"

                new_zoom *= ((1 + (last_two_thumb_index_dists[1] - last_two_thumb_index_dists[0]))) ** (1/ZOOM_SENSITIVITY) # outer plus sign bc pinch out means zoom in
                return display_message, new_zoom

            if hand_status == ['Left']*MIN_WAITING_FRAMES and \
                hand_open(middle_finger_open_list, MIN_WAITING_FRAMES):

                # Calculate rotation matrix and extract angles

                display_message = "Rotating"

                normal_to_rotate = np.cross(np.average(last_two_thumb_index_vecs,axis=0), z_unit_vec) # always crossing it into the screen..check sign later
                angle_to_rotate = np.average(last_two_thumb_index_dists)

                return display_message, [normal_to_rotate[::-1], angle_to_rotate*ROTATION_SENSITIVITY]
            # print(hand_status)
    else: # i.e. no hands detected
                                            
        thumb_positions = MaxSizeList(MAX_TRACKING_TIME) 
        index_positions = MaxSizeList(MAX_TRACKING_TIME)
        
        pause_updates = True
        
    # Show vtk file and camera's image
    if pause_updates:
        display_message = "Updates paused"

    return display_message, 0



from PyQt5.QtCore import QRunnable, Qt, QThreadPool
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFrame
)
# 1. Subclass QRunnable
# class Runnable(QRunnable):
#     def __init__(self, capture):
#         super().__init__()
#         self.n = n

#     def run(self):
#         # Your long-running task goes here ...
#         while self.capture.isOpened():
#             with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85, max_num_hands=2) as hands_object:
#                 display, remainder = camera_loop(capture, hands_object, hand_status, thumb_positions,index_positions,middle_tip_vert_positions, middle_palm_vert_positions, middle_finger_open_list, last_two_positions)
                    

import sys
import os
import socket
from PyQt5 import QtCore, QtWidgets


class UDPWorker(QtCore.QObject):
    dataChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(UDPWorker, self).__init__(parent)

    @QtCore.pyqtSlot()
    def start(self, capture):
        self.capture = capture
        self.process()

    def process(self):
        while self.capture.isOpened():
            with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85, max_num_hands=2) as hands_object:
                display, remainder = camera_loop(self.capture, hands_object, hand_status, thumb_positions,index_positions,middle_tip_vert_positions, middle_palm_vert_positions, middle_finger_open_list, last_two_positions)
                    
                self.dataChanged.emit(display) # ,remainder
                print(display)




class MainWindow(QMainWindow):
    started = QtCore.pyqtSignal()

    def __init__(self, stl_file, parent=None, capture=cv2.VideoCapture(0)):
        super(MainWindow, self).__init__(parent)

        QMainWindow.__init__(self, parent)
        self.frame = QFrame()
        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.capture = capture
        self.display = ''
        self.remainder = 0
        vp = Plotter(qtWidget=self.vtkWidget)
        self.model = Mesh(stl_file)

        vp.show(self.model, viewup="z", interactorStyle=0, camera=dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1)))
        self.start(vp)

  
    def start(self, vp):

        for r in vp.renderers:
            self.vtkWidget.GetRenderWindow().AddRenderer(r)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()


        def keypress(obj, e):
            vp._keypress(obj, e)
            if self.iren.GetKeySym() in ["q", "space"]:
                self.iren.ExitCallback()
                exit()

        self.iren.AddObserver("KeyPressEvent", keypress)


        

        # vp.show(self.model, viewup="z", interactorStyle=0, camera=dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1)))

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show()  # qt not Plotter method
        r.ResetCamera()
        self.iren.Start()


    @QtCore.pyqtSlot(str)
    def addItem(self, text):
        # self.lst.insertItem(0, text)
        self.display = text
        

    def onClose(self):
        print("Disable the interactor before closing to prevent it from trying to act on already deleted items")
        self.vtkWidget.close()























STL= r'C:\Users\jacob\Downloads\croc-nut20210722-6981-17x5gmo\rayandsumer\croc-nut\cn1.stl'


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(STL)
    worker = UDPWorker()
    thread = QtCore.QThread()
    thread.start()
    worker.moveToThread(thread)
    window.started.connect(worker.start)
    worker.dataChanged.connect(window.addItem)
    window.show()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()

