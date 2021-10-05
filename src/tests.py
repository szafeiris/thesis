from AVClassifier import variables as v
from AVClassifier import preprocessing
from PIL import Image

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

    imagePath = '../data/media/selected_frames/bicuspid/1/frame35.jpg'
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



if __name__ == "__main__":
    totalTests = len(TESTS)
    print(f'Executing {totalTests} tests...\n')
    failedTests = 0
    for test in TESTS:
        if test() == TEST_FAILURE:
            failedTests += 1
    print(f'\nTesting completed: {failedTests} of {totalTests} test(s) failed, {totalTests-failedTests} test(s) succeed.\n\n')