from AVClassifier import variables as v
from AVClassifier import preprocessing
from AVClassifier import augmentation
from PIL import Image
import numpy as np

TESTS = []
TEST_SUCCESS = 0
TEST_FAILURE = 1

def fTest(testName, errorMsg):
    print(f'> Test "{testName}" failed. {errorMsg}.')

def sTest(testName):
    print(f'> Test "{testName}" executed successfully.')

### Tests definition

## Preprocessing
def preprocessingTest():
    testname = 'preprocessing'

    imagePath = v.DATA_PATH + 'bicuspid/1/frame35.jpg'
    image = Image.open(imagePath)
    pimage = preprocessing.imagePreprocessor( image, 
                                              cropDim=v.ABNORMAL_CROP_DIM,
                                              resizeDim=v.RESIZE_DIM
                                            )
    
    if pimage.size != v.RESIZE_DIM:
        fTest(testname, f"Image final size does not match {v.RESIZE_DIM}")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(preprocessingTest)


## Augmentation
def augmentationNoiseGenTest():
    testname = 'augmentationNoiseGen'

    aimage = augmentation.generateImageNoise(v.RESIZE_DIM)
    
    if aimage.shape != v.RESIZE_DIM:
        fTest(testname, f"Image final shape does not match {v.RESIZE_DIM}")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(augmentationNoiseGenTest)

def augmentationNoiseTest():
    testname = 'augmentationNoise'

    imagePath = v.DATA_PATH + 'bicuspid/1/frame35.jpg'
    image = Image.open(imagePath)
    pimage = preprocessing.imagePreprocessor( image, 
                                              cropDim=v.ABNORMAL_CROP_DIM,
                                              resizeDim=v.RESIZE_DIM
                                            )
    pimage = np.asarray(pimage)
    aimage = augmentation.addNoise(pimage)

    if np.array_equal(pimage, aimage):
        fTest(testname, f"Original image have not been altered")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(augmentationNoiseTest)

def augmentationHFlipTest():
    testname = 'augmentationHFlip'

    imagePath = v.DATA_PATH + 'bicuspid/1/frame35.jpg'
    image = Image.open(imagePath)
    pimage = preprocessing.imagePreprocessor( image, 
                                              cropDim=v.ABNORMAL_CROP_DIM,
                                              resizeDim=v.RESIZE_DIM
                                            )
    pimage = np.asarray(pimage)
    aimage = augmentation.horizontalFlip(pimage)

    if np.array_equal(pimage, aimage):
        fTest(testname, f"Original image have not been altered")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(augmentationHFlipTest)

def augmentationJitterTest():
    testname = 'augmentationJitter'

    imagePath = v.DATA_PATH + 'bicuspid/1/frame35.jpg'
    image = Image.open(imagePath)
    pimage = preprocessing.imagePreprocessor( image, 
                                              cropDim=v.ABNORMAL_CROP_DIM,
                                              resizeDim=v.RESIZE_DIM
                                            )
    pimage = np.asarray(pimage)
    aimage = augmentation.jitterImage(pimage)

    if np.array_equal(pimage, aimage):
        fTest(testname, f"Original image have not been altered")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(augmentationJitterTest)

def augmentationTranslateTest():
    testname = 'augmentationTranslate'

    imagePath = v.DATA_PATH + 'bicuspid/1/frame35.jpg'
    image = Image.open(imagePath)
    pimage = preprocessing.imagePreprocessor( image, 
                                              cropDim=v.ABNORMAL_CROP_DIM,
                                              resizeDim=v.RESIZE_DIM
                                            )
    pimage = np.asarray(pimage)
    aimage = augmentation.translateImage(pimage, prob=np.random.choice(v.TRANSLATE_PROBS))

    if np.array_equal(pimage, aimage):
        fTest(testname, f"Original image have not been altered")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(augmentationTranslateTest)

def augmentationShearTest():
    testname = 'augmentationTranslate'

    imagePath = v.DATA_PATH + 'bicuspid/1/frame35.jpg'
    image = Image.open(imagePath)
    pimage = preprocessing.imagePreprocessor( image, 
                                              cropDim=v.ABNORMAL_CROP_DIM,
                                              resizeDim=v.RESIZE_DIM
                                            )
    pimage = np.asarray(pimage)
    aimage = augmentation.shearImage(pimage, prob=np.random.choice(v.TRANSLATE_PROBS))

    if np.array_equal(pimage, aimage):
        fTest(testname, f"Original image have not been altered")
        return TEST_FAILURE
    else:
        sTest(testname)
        return TEST_SUCCESS

TESTS.append(augmentationShearTest)


if __name__ == "__main__":
    totalTests = len(TESTS)
    print(f'Executing {totalTests} test(s)...\n')
    failedTests = 0
    for test in TESTS:
        if test() == TEST_FAILURE:
            failedTests += 1
    print(f'\nTesting completed: {failedTests} of {totalTests} test(s) failed, {totalTests-failedTests} test(s) succeed.\n\n')