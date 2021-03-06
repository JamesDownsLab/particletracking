import cv2 

EXAMPLE_CHILD_PARAMETERS = {
    'crop method': 'manual',
    'no_of_sides': 6,
    'method': ('flip', 'crop_and_mask', 'grayscale', 'threshold'),
    'threshold': [50, 1, 255, 1],
    'threshold mode': cv2.THRESH_BINARY_INV,
    'min_dist': [23, 3, 51, 1],
    'p_1': [105, 1, 255, 1],
    'p_2': [2, 1, 20, 1],
    'min_rad': [13, 1, 101, 1],
    'max_rad': [14, 1, 101, 1],
    'max frame displacement': 10,
    'min frame life': 5,
    'memory': 3
    }