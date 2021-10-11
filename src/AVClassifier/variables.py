## General purpose
CLASSES = ['tricuspid', 'bicuspid', 'raphe']
BASE_PATH = '../data/media/'
DATA_PATH = '../data/media/selected_frames/'

VIDEO_PATH = f'{BASE_PATH}/selected_frames/'
IMAGE_PATH = f'{BASE_PATH}/original/images/'

VIDEO_NPY_PATH = f'{BASE_PATH}/npy_files/videos/'
IMAGE_NPY_PATH = f'{BASE_PATH}/npy_files/images/'


## Pre-processing variables
ABNORMAL_CROP_DIM = (1016//2 - 671//2, 708//2 - 612//2, 1016//2 + 671//2, 708//2 + 612//2)
NORMAL_CROP_DIM = (68, 5, 570, 386)
RESIZE_DIM = (224, 224)

## Augmentetion variables
TRANSLATE_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHEAR_PROBS = [0.3, 0.4, 0.5, 0.6]