class FeatureExtraction:

    def __init__(self, extractorIN, param):
        self.extractor = extractorIN.featureExtraction(param)


    def extract(self, images):
        x = self.extractor.runAll(images)
        return x