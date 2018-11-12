import cv2
import os

class ImageFinder:

    def __init__(self, location, classForms, size=None):
        self.directory = location
        self.x = []
        self.y = []
        self.classForms = classForms
        self.size = size

    def returnClasses(self):
        for root, dirs, filenames in os.walk(self.directory):
            for f in filenames:
                for cForm in self.classForms:
                    if cForm in f:
                        loc = os.path.join(root, f)
                        img = cv2.imread(loc, 0)
                        self.x.append(img)
                        self.y.append(cForm)
        return self.x, self.y