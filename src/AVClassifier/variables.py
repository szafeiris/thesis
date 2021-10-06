## General purpose
CLASSES = ['tricuspid', 'bicuspid', 'raphe']
DATA_PATH = '../data/media/selected_frames/'

## Pre-processing variables
ABNORMAL_CROP_DIM = (1016//2 - 671//2, 708//2 - 612//2, 1016//2 + 671//2, 708//2 + 612//2)
NORMAL_CROP_DIM = (68, 5, 570, 386)
RESIZE_DIM = (224, 224)

## Augmentetion variables
TRANSLATE_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHEAR_PROBS = [0.3, 0.4, 0.5, 0.6]