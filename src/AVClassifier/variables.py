## General purpose
CLASSES = ['tricuspid', 'bicuspid', 'raphe']
DATA_PATH = '../data/media/selected_frames/'
MODELS_DATA_PATH = '../data/models/'

## Pre-processing variables
ABNORMAL_CROP_DIM = (1016//2 - 671//2, 708//2 - 612//2, 1016//2 + 671//2, 708//2 + 612//2)
NORMAL_CROP_DIM = (68, 5, 570, 386)
RESIZE_DIM = (224, 224)

## Augmentetion variables
TRANSLATE_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SHEAR_PROBS = [0.3, 0.4, 0.5, 0.6]
TRANSLATE_RANGE = 5
SHEAR_RANGE = 10

## Model variables
EPOCHS = 100
BATCH_SIZE = 1
USE_BATCH_NORM = True
USE_DROPOUT = False
BASE_FILTERS = 32
DROPOUT_PROBS = [0.5, 0.5]
LEARNING_RATE = 0.00001
DECAY =  1e-6
LOSS_FUN = 'categorical_crossentropy'
INPUT_DATA_SHAPE_3D = (40, 224, 224, 1)

TRAIN_SHUFFLE = True
USE_MP = False
WORKERS = 6
USE_GPU = True
VERBOSE = 1

## Data variables
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO  = 0.15