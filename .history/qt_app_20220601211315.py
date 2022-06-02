from qt_ex2 import camera_loop
import sys
import os
import socket
from PyQt5 import QtCore, QtWidgets, QtGui
import time
import numpy as np

"""
A basic PyQt-based GUI rendering of handtracking-based 3D file manipulation; A.K.A. N(o)H(ands)3D
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
PAN_SENSITIVITY = 2
PAN_Z_SENSITIVITY = 10
ZOOM_SENSITIVITY = 0.1 # effectively how many loop iterations must be done (i.e. ms waited) to acheive zoom factor
INITIAL_RESCALE = 15

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
                index_change = [index_change[2], index_change[0], -index_change[1]]

                return display_message, 10 * PAN_SENSITIVITY * np.array(index_change)
                
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
    # cv2.imshow('MediaPipe Hand Detection', image)

    return display_message, 0

class ParallelWorker(QtCore.QObject):
    dataChanged = QtCore.pyqtSignal(str, list)

    def __init__(self, parent=None):
        super(ParallelWorker, self).__init__(parent)
        self.server_start = False
        self.capture = None

    @QtCore.pyqtSlot()
    def start(self):
        self.server_start = True
        self.capture=cv2.VideoCapture(0)

        self.process()

    def process(self):
        with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85, max_num_hands=2) as hands_object:
            while self.server_start: # self.capture.isOpened()
                display, remainder = camera_loop(self.capture, hands_object, hand_status, thumb_positions,index_positions,middle_tip_vert_positions, middle_palm_vert_positions, middle_finger_open_list, last_two_positions)
                
                self.dataChanged.emit(str(display), [remainder]) # to send thru the proper type

    def onClose(self):
        self.capture.release()
        # cv2.destroyAllWindows()

class DataStream():

    def __init__(self, model, vp):
        self.model = model
        # self.camera = cam
        self.vp = vp
        self.keypress = self.vp.addCallback("key press",   self.onKeypress)

    @property
    def value(self):
        """I'm the 'value' property."""
        # print("getter of x called")
        return self._value

    @value.setter
    def value(self, new_val):
        # print(new_val)
        self._value = new_val

    def updateModel(self, ROTATION_SENSITIVITY, ZOOM_SENSITIVITY, PAN_SENSITIVITY):

        self.disp, self.remain = self.value
        self.ROTATION_SENSITIVITY = ROTATION_SENSITIVITY
        self.ZOOM_SENSITIVITY = ZOOM_SENSITIVITY
        self.PAN_SENSITIVITY = PAN_SENSITIVITY

        if self.disp == 'Rotating':
            # print(self.remain)
            rot_axis, rot_angle = self.remain[0]
            self.model.rotate(axis=rot_axis, angle = ROTATION_SENSITIVITY * rot_angle)
            self.vp.interactor.Render()

             
        if self.disp == 'Zooming':
            zoom_ratio = self.remain[0]
            self.model.scale(ZOOM_SENSITIVITY * zoom_ratio)
            self.vp.interactor.Render()

            
        if self.disp == 'Panning':
            panning_vector = self.remain[0]
            self.model.shift(PAN_SENSITIVITY * panning_vector )
            self.vp.interactor.Render()


        if self.disp == 'Updates paused':
            pass

        if self.disp == 'Resetting':
            self.vp.plt.resetCamera()
            self.model = Mesh(self.model.filename)

    def onKeypress(self, evt):
        if evt.keyPressed == 'r':
            print('in onkeypress')
            self.disp = 'Resetting' #Mesh(self.model.filename)
    # def onKeypress(self, evt):
    #     from vedo import printc
    #     printc("You have pressed key:", evt.keyPressed, c='b')
    

cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout,
                             QGroupBox,QMenu, QPushButton,
                             QRadioButton, QVBoxLayout,
                             QWidget, QSlider,QLabel, QAbstractSlider)

from PyQt5.QtCore import Qt as Qtc

class CustomSlider(QtWidgets.QWidget):
    def __init__(self, orientation, control_name: str, control_default: int, control_min: int, control_max: int, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.control_name = control_name

        if self.orientation in {'vertical', 'v', 'Vertical', 'VERTICAL'}:
            self.slider = QtWidgets.QSlider(orientation=QtCore.Qt.Vertical)
        if self.orientation in {'horizontal', 'h', 'Horizontal', 'HORIZONTAL'}:
            self.slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        else:
            print(f'{self.orientation} not in available orientations.')
            raise TypeError


        self.control_variable = control_default

        self.slider.setRange(control_min, control_max)
        self.slider.setFocusPolicy(Qtc.NoFocus)
        self.slider.setPageStep(5)
        self.slider.setValue(self.control_variable)
        self.slider.valueChanged.connect(self.updateLabel)

        self.label = QLabel(f'{self.control_name}: {str(self.slider.value())}', self)
        self.label.setAlignment(Qtc.AlignCenter | Qtc.AlignVCenter)
        self.label.setMinimumWidth(80)
        self.label.setMaximumWidth(1000)

    def updateLabel(self, value):
        self.label.setText(f'{self.control_name}: {str(value)}')
        self.control_variable = int(value)
        



class ParentWidget(Qt.QMainWindow):
    started = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)
       

        self.filename = ''
        self.display = ''
        self.remainder = None
        self.start_vid = False

        self.setWindowTitle(f'3DM')
        
        self.setupUI()
        self.showMaximized()

    def setupUI(self):
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.q_label = QtWidgets.QLabel('Firing up...')
        self.vl.addWidget(self.q_label) 

        self.choose_file_button = QtWidgets.QPushButton("Select file")
        self.choose_file_button.clicked.connect(self.file_open)
        self.vl.addWidget(self.choose_file_button)
                        
        self.start_gestures_button = QtWidgets.QRadioButton("Begin Video Feed")
        self.start_gestures_button.setCheckable(True)
        self.start_gestures_button.clicked.connect(self.started)
        # print(self.start_gestures_button.isChecked())
        self.vl.addWidget(self.start_gestures_button)     
 
        self.setWindowTitle(f"3DM - {self.filename}")

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.rotation_slider = CustomSlider('Horizontal', 'Rotation Sensitivity', ROTATION_SENSITIVITY, 1, 20)
        self.zoom_slider = CustomSlider('Horizontal', 'Zoom Sensitivity', ZOOM_SENSITIVITY, 1, 20)
        self.pan_slider = CustomSlider('Horizontal', 'Pan Sensitivity', PAN_SENSITIVITY, 1, 20)


        for slide_obj in [self.rotation_slider, self.zoom_slider, self.pan_slider]:
            self.vl.addWidget(slide_obj.slider)
            self.vl.addSpacing(15)
            self.vl.addWidget(slide_obj.label)


        self.setGeometry(300, 300, 350, 250)


    def file_open(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]
        # Create renderer and add the vedo objects and callbacks
        self.vp = Plotter(qtWidget=self.vtkWidget, interactive=True)
        # self.vp.backgroundColor('black')
        # self.id1 = self.vp.addCallback("mouse click", self.onMouseClick)
        self.model = Mesh(self.filename)
        self.avg_model_size = self.model.averageSize()
        cam['pos'] = (self.avg_model_size*INITIAL_RESCALE,0,0)

        # self.model.scale(self.avg_model_size * INITIAL_RESCALE)
        self.vp.show(self.model, camera = cam)
        self.lst = DataStream(self.model, self.vp)

        

    @QtCore.pyqtSlot(bool)  #<<== the missing link
    def on_pushButtonSetBase_toggled(self, checked):
        if checked:
            self.rowOverride = True
        elif not checked:
            self.rowOverride = False

    @QtCore.pyqtSlot(str, list)
    def addItem(self, text, remainder):
        # self.lst.setText(0, text)
        self.lst.value = text, remainder
        self.lst.updateModel(
            self.rotation_slider.slider.value(), self.zoom_slider.slider.value(), self.pan_slider.slider.value()
            )
        self.q_label.setText(self.lst.value[0])
        if self.lst.value[0] != 'Updates paused':
            self.vp.show(self.model, camera = cam, interactive = True)                  # <--- show the vedo rendering


    def onClose(self):
        """
        Safely close vtk window and camera! Helps prevent glitches when exiting or crashing
        """
        self.vtkWidget.close()
        # self.started.capture.close


STL = r'C:\Users\jacob\Downloads\croc-nut20210722-6981-17x5gmo\rayandsumer\croc-nut\cn1.stl'

from styles import style_sheet
import qdarkstyle
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyleSheet(style_sheet)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    window = ParentWidget()
    # Start a parallel thread worker to simultaneously loop thru camera images and perform hand calculations
    camera_worker = ParallelWorker()
    thread = QtCore.QThread()
    thread.start()
    camera_worker.moveToThread(thread)
    # Parallel thread ^

    # Connect window.started to camera_worker.start
    window.started.connect(camera_worker.start)
    camera_worker.dataChanged.connect(window.addItem)
    # send data from camera_worker to the main window via addItem

    window.show()
    
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    sys.exit(app.exec_())
