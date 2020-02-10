from labvision import images 
import numpy as np
import cv2 


def manual(frame, parameters):
    no_of_sides = parameters['No of sides']
    if no_of_sides == 1:
        crop_result = images.crop_circle(frame)
    else:
        crop_result = images.crop_polygon(frame)

    return crop_result.bbox, crop_result.mask, crop_result.points


def nocrop(frame, parameters):
    w, h = images.width_and_height(frame)
    return None, None, np.array([[0, 0], [0, w], [w, h], [0, h]])
