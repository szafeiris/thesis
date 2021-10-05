## General purpose
CLASSES = ['tricuspid', 'bicuspid', 'raphe']

## Pre-processing variables
ABNORMAL_CROP_DIM = (1016//2 - 671//2, 708//2 - 612//2, 1016//2 + 671//2, 708//2 + 612//2)
NORMAL_CROP_DIM = (68, 5, 570, 386)
RESIZE_DIM = (224, 224)