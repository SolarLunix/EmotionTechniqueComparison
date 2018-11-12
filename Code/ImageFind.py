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
                        if self.size != None:
                            img = self.croptoface(img)
                        self.x.append(img)
                        self.y.append(cForm)
        return self.x, self.y

    def croptoface(self, img):
        faceD = cv2.CascadeClassifier('Assets\Cascader\haarcascade_frontalface_default.xml')
        face = faceD.detectMultiScale(
            img,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if face != ():
            x, y, w, h = face[0]
            new_img = img[y:y + h, x:x + w]
        else:
            new_img = img

        new_img = cv2.resize(new_img, (130, 130))
        return new_img