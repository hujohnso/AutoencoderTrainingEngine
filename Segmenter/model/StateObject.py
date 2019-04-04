class StateObject:
    # 50 might be too large.  I am not sure how quickly the centroids will change direction
    # centroids = []

    # This is for if the labeler doesn't label the object with the same label as was before
    def __init__(self, object_id):
        self.object_id = object_id
        self.current_mapped_number = object_id
        self.number_of_times_recorded = 0
        self.current_pixels = []
        self.current_momentum = None
        self.folder_path = None
        self.current_centroid = None

    def set_current_pixels(self, current_pixels):
        self.current_pixels = current_pixels

    def get_object_id(self):
        return self.object_id

    def get_number_of_times_recorded(self):
        return self.number_of_times_recorded

    def advance_number_of_times_recorded(self):
        self.number_of_times_recorded += 1

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path

    def get_folder_path(self):
        return self.folder_path

    def set_new_mapping(self, folder_path):
        self.folder_path = folder_path

    def get_current_mapping(self):
        return self.folder_path

    def set_current_centroid(self, new_centroid):
        self.current_centroid = new_centroid
        #self.centroids.append(new_centroid)





