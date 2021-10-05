from PIL import Image

def imagePreprocessor(image, cropDim=None, resizeDim=None, doGray=True):
    if cropDim is not None:
        image = image.crop(cropDim)
    
    if resizeDim is not None:
        # Use a high-quality downsampling filter (ANTIALIAS)
        image = image.resize(resizeDim, Image.ANTIALIAS)
            
    if doGray:
        image = image.convert('L') # Luma transform
    
    return image
