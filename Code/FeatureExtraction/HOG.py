from skimage.feature import hog

class featureExtraction():

    def __init__(self, params):
        self.params = params

    def runAll(self, images):
        for i, img in enumerate(images):
            images[i] = hog(img, self.params.get('orientation'), self.params.get('ppc'), self.params.get('cpb'), self.params.get('norm'))
        return images