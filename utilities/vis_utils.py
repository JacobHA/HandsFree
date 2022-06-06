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



