import cv2
import os

class ckPlus():

    def __init__(self):
        self.features = []
        self.labels = []

    def extractImages(self,directory):
        neutralPerson = None
        for root, dirs, filenames in os.walk(directory):
            for f in filenames:
                loc = os.path.join(root, f)
                #file = loc + f
                emotionTypeFile = open(loc, 'r')
                emotionType = emotionTypeFile.readline()
                emotionType = float(emotionType.split()[0])
                #emotionType = (intemotionType
                structure = root.split('\\')
                if(structure[-2] != neutralPerson):
                    imgFileLoc = 'cohn-kanade-images'.join(loc.rsplit('Emotion', 1))
                    imgFileLoc = imgFileLoc.replace('_emotion.txt', '.png')
                    neutralImage = imgFileLoc[0:-6]
                    neutralImage += '01.png'
                    img = cv2.imread(neutralImage, 0)
                    self.features.insert(len(self.features), img)
                    self.labels.insert(len(self.labels), 'Neutral')
                    neutralPerson = structure[-2]

                if(emotionType == 0.0):
                    self.addEmotionImage(loc,'Neutral')
                elif(emotionType == 1.0):
                    self.addEmotionImage(loc,'Angry')
                elif(emotionType == 2.0):
                    self.addEmotionImage(loc,'Contempt')
                elif(emotionType == 3.0):
                    self.addEmotionImage(loc,'Disgust')
                elif(emotionType == 4.0):
                    self.addEmotionImage(loc,'Fear')
                elif(emotionType == 5.0):
                    self.addEmotionImage(loc,'Happy')
                elif(emotionType == 6.0):
                    self.addEmotionImage(loc,'Sad')
                elif(emotionType == 7.0):
                    self.addEmotionImage(loc,'Surprise')

                #img = cv2.imread(loc, 0)
                # img = Image.open(loc).convert('L')
                # img = img_as_ubyte(img)



    def addEmotionImage(self,imgFileLoc,emotion):
        #imgFileLoc = imgFileLoc.replace('Emotion','cohn-kanade-images')
        imgFileLoc = 'cohn-kanade-images'.join(imgFileLoc.rsplit('Emotion', 1))
        imgFileLoc = imgFileLoc.replace('_emotion.txt','.png')
        ##imgFileLoc = imgFileLoc.replace('\\','/')
        img = cv2.imread(imgFileLoc,0)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        self.features.insert(len(self.features), img)
        self.labels.insert(len(self.labels), emotion)

    def getFeatures_and_Labels(self):
        return self.features, self.labels