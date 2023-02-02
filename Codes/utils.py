import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import string
import random
import cv2 as cv

class Utils():
    def __init__(self,vocabSize = 2000):
        self.vocabSize = vocabSize
        self.trainSplit = 0.7
        self.testSplit = 0.2
        self.validationSplit = 0.1
        self.Data = []
        self.img = []
        pass
    def readCSV(self):
        man = pd.read_csv(r"C:\Users\Legion\Documents\AI\Git\Image captioning\Data\images\image_captions.csv")
        self.imgName = man['img_name']
        self.imgCaption = '[Start] ' + man['img_caption'] + ' [End]'
    def loadImages(self):
        for i,im in enumerate(self.imgName):
            img = cv.imread(f"C:\\Users\\Legion\\Documents\\AI\\Git\\Image captioning\\Data\\images\\{im}")
            img = img/255.0
            img = cv.resize(img,[256,256])
            self.img.append(img)
        self.img = np.array(self.img)
    def corpusPreprocessing(self):
        seqLen = 10
        self.textVectorization = tf.keras.layers.TextVectorization(max_tokens = self.vocabSize,
                                                                    output_mode = 'int',
                                                                    output_sequence_length = seqLen,
                                                                    standardize = self.standardation)
        self.textVectorization.adapt(self.imgCaption.to_numpy())
        self.vectorizedCaptions = self.textVectorization(self.imgCaption)
    def trainTestSplit(self):
        for i in range(len(self.img)):
            self.Data.append([self.img[i],self.vectorizedCaptions[i][:-1],self.vectorizedCaptions[1:]])
        random.shuffle(self.Data)
        numValSamples = int(self.validationSplit * len(self.Data))
        numTrainSamples = int(self.trainSplit*len(self.Data))
        trainData = self.Data[:numTrainSamples]
        valData = self.Data[numTrainSamples:numValSamples + numTrainSamples]
        testData = self.Data[numValSamples + numTrainSamples:]
        return [trainData,valData,testData]
    @staticmethod
    def standardation(inputString):
        punc = string.punctuation
        punc = punc.replace('[','')
        punc = punc.replace(']','')
        inputString = tf.strings.lower(inputString)
        inputString = tf.strings.regex_replace(inputString,f'[{punc}]','')
        return inputString