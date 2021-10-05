import numpy as np
import cv2

def generateImageNoise(size):
    # Generate Gaussian noise
    noise = np.random.randint(0, 45, size)
    noise = noise.reshape(size[0], size[1]).astype('uint8')

    return noise


def addNoise(image, noise=None):
    if noise is None:
        # Generate Gaussian noise
        noise = np.random.randint(0, 45, image.size)
        noise = noise.reshape(image.shape[0], image.shape[1]).astype('uint8')

    # Add the Gaussian noise to the image
    imageNoise = cv2.add(image, noise)

    return imageNoise


def horizontalFlip(image):
    return cv2.flip(image, 1)


def jitterImage(image):
    if np.random.uniform(low=0, high=1, size=(1,))[0] > 0.5:
        image_out = np.clip(image + image*0.25, 0,255)
    else:
        image_out = np.clip(image - image*0.25, 0,255)
    #image_out =  np.clip(image + 4, 0,255)
    return image_out.astype('uint8')


def translateImage(image, range = 5, prob=np.random.uniform()):
    rows, cols = image.shape

    tr_x = range*prob-range/2
    tr_y = range*prob-range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    image = cv2.warpAffine(image, Trans_M,(cols,rows))

    return image


def shearImage(image, range = 10, prob=np.random.uniform()):
    rows, cols = image.shape    

    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+range*prob-range/2
    pt2 = 20+range*prob-range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)

    image = cv2.warpAffine(image, shear_M,(cols,rows))

    return image
