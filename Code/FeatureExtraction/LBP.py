from skimage.feature import multiblock_lbp as lbp
import numpy as np

name = "LBP"

class featureExtraction():

    def __init__(self, params):
        self.params = params

    def runAll(self, images):
        for i, img in enumerate(images):
            images[i] = lbp(img, self.params.get('r'), self.params.get('c'), self.params.get('width'), self.params.get('height'))
            images[i] = np.array(images[i]).flatten()
        return images