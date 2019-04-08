class StateObjectClasses:
    def __init__(self, array_of_state_object_images, array_of_state_object_classes):
        self.array_of_state_object_images = array_of_state_object_images
        self.array_of_state_object_classes = array_of_state_object_classes
        self.number_of_objects_identified = array_of_state_object_classes.shape[1]
