from skimage.feature import local_binary_pattern as lbp
import scipy
import cv2
import numpy as np

name = "LBP"

class featureExtraction():

    def __init__(self, params):
        self.params = params

    def runAll(self, images):
        for i, img in enumerate(images):
            myimg = lbp(img, self.params.get('p'), self.params.get('r'), self.params.get('method'))

            hist, _ = np.histogram(myimg.ravel(), bins=np.arange(0, self.params.get('p') + 3), range=(0, self.params.get('p') + 2))
            hist = hist.astype("float")
            eps = 1e-7
            hist /= (hist.sum() + eps)
            images[i] = np.array(hist).flatten()
        return images