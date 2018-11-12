import cv2

def croptoface(img):
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
