import numpy as np
import cv2 
from labvision import images 
from particletracking.preprocessing import methods, crops 


class PreProcessor:
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.crop_method = self.parameters['crop method']
        self.mask_im = np.array([])
        self.crop = []
        self.boundary = None
        self.calls = 0
        
    def update_parameteres(self, parameters):
        self.parameters = parameters
        
    def process(self, frame):
        if self.calls == 0:
            self.crop, self.mask_im, self.boundary = getattr(
                crops, self.crop_method)(frame, self.parameters)
            self.parameters['crop'] = self.crop
            self.parameters['mask image'] = self.mask_im
            
        if 'crop_and_mask' not in self.parameters["method"]:
            cropped_frame = frame.copy()

        # perform each method in the method list
        for method in self.parameters["method"]:
            frame = getattr(methods, method)(frame, self.parameters)
            if method == "crop_and_mask":
                cropped_frame = frame.copy()
        
        self.calls += 1
        return frame, self.boundary, cropped_frame