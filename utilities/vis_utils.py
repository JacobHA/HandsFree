from vedo import *
from vedo.assembly import Assembly

from utilities.data_utils import is_within


cam = dict(pos=(1,0,0), focalPoint=(0,0,0), viewup=(0,0,1))

class DisplayScene:
    def __init__(self, camera=cam, initial_msg='Firing up...', axes=4, init_scale=1):
        self.axes = axes
        self.init_scale = init_scale
        self.camera = camera
        self.status_message = Text2D(initial_msg, pos="top-center", font=2, c='w', bg='b3', alpha=1)
        self.objects = []
        self.active_objects = []
        self.planes = []
        self.assembly = None
        self.selector = Point((0,0,0), alpha=0)

        self.initial_display()

    # Update functions
    def update_message(self, message):
        self.status_message.text(message)
        
    # Display functions
    def initial_display(self):
        # Display the scene with initialized objects at origin
        if len(self.objects) > 0:
            for obj in self.objects:
                obj.initial_setup()
            _ = show([obj.mesh for obj in self.active_objects], self.status_message, axes=self.axes, viewup='z', camera=self.camera, interactive=False)
        
        if len(self.objects) == 0:
            _ = show(self.status_message, axes=self.axes, viewup='z', camera=self.camera, interactive=False)

    def display_objects(self):
        # Only display the unlocked objects 
        # We will create a new mesh by merging all the unlocked meshes
        # self.create_assembly()
        _ = show([o.mesh for o in self.objects], self.planes, self.selector, self.status_message, axes=self.axes, viewup='z', camera=self.camera, interactive=False)
        

    # Object functions
    def add_object(self, new_object):
        if new_object is not None:
            # assert type(new_object) == ManipulableObject, 'Input object must be of type: ManipulableObject.'
            self.active_objects.append(new_object)
            self.objects.append(new_object)
    
    # def create_assembly(self):
    #     # Update the assembly object    
    #     total = Mesh()
    #     for obj in self.objects:
    #         if not obj.is_locked:
    #             total += obj.mesh
    #     self.assembly = ManipulableObject(mesh=merge([o.mesh for o in self.objects]))

    # Selector functions
    def show_selector(self, loc_vec):
        self.selector.alpha = 1
        self.selector.addPos(loc_vec)

    def select_object(self):
        for obj_num, obj in enumerate(self.objects):
            x=obj.mesh.clone().projectOnPlane('x').silhouette('2d').points()
            point_location = self.selector.pos()
            if is_within(x, point_location):
                obj.lock()
                # Color the object's mesh
                print("LOCKING")
                self.update_message(f'Selected Object #{obj_num}')

                            
    def show_cross_section(self, axis='y', point=None):
        for obj in self.objects:
            if point is None:
                point = obj.mesh.centerOfMass()
            cut_mesh, plane = obj.cross_section(axis, point=point)
            self.objects.remove(obj)
            self.add_object(cut_mesh)
            self.planes.append(plane)

    def remove_selector(self):
        self.selector.alpha = 0

# TODO:
# Make a parent class, with Mesh and Volume objects as children
class ManipulableObject:
    def __init__(self, filename=None, mesh=None, initial_scale=1):
        self.is_locked = False
        self.filename = filename
        self.initial_scale = initial_scale
        if mesh is None: self.mesh = Mesh(filename)
        else: self.mesh = mesh

        self.initial_setup()

    # Initial functions
    def initial_setup(self):
        # Do fresh import of object
        self.mesh = self.mesh.lighting('plastic').color('b')
        self.mesh.scale(self.initial_scale * self.mesh.averageSize())
        # self.mesh.shift(-self.mesh.centerOfMass())

    # Update/Manipulation functions
    def shift_object(self, shift_vector):
        if not self.is_locked:
            self.mesh.shift(shift_vector)

    def scale_object(self, scale):
        if not self.is_locked:
            # Normalize by average size:
            self.mesh.scale( scale )

    def rotate_object(self, angle, axis, point=None):
        if not self.is_locked:
            if point is None:
                point = self.mesh.centerOfMass()
                self.mesh.rotate(angle=angle, axis=axis, point=point)

    # Keyboard functions:
    def lock(self):
        self.is_locked = True
        self.lock_recolor()
    def swap_lock(self):
        self.is_locked = not self.is_locked
        self.lock_recolor()
    def unlock(self):
        self.is_locked = False
        self.lock_recolor()
    def lock_recolor(self):
        if self.is_locked:
            self.mesh.color('red')
        if not self.is_locked:
            self.mesh.color('blue')

    # Cross-section functions:
    def cross_section(self, axis, point=None):
        if point is None:
            _,_, y0, y1, z0, z1 = self.mesh.bounds()
            point = (0,y1*0.99,0)
        if axis == 'y':
            axis = [0,1,0]
        cut_mesh = self.mesh.clone().cutWithPlane(origin=point, normal=axis)
            # v.cutWithPlane(origin=orig, normal=[0,1,0])
        

        pl = Plane(point, normal=axis, s=[self.mesh.averageSize()]*2, c='black', alpha=0.7)

        return ManipulableObject(mesh=cut_mesh), pl

