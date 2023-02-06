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
        self.encoderInput = '[Start] ' + man['img_caption']
        self.encoderOutput =  man['img_caption'] + ' [End]'
    def loadImages(self):
        for im in self.imgName:
            img = cv.imread(f"C:\\Users\\Legion\\Documents\\AI\\Git\\Image captioning\\Data\\images\\{im}")
            img = img/255.0
            img = cv.resize(img,[256,256])
            self.img.append(np.expand_dims(img,axis = 0))
        self.img = np.array(self.img)
        self.img = np.concatenate(self.img,axis = 0)
    def corpusPreprocessing(self):
        seqLen = 10
        self.textVectorization = tf.keras.layers.TextVectorization(max_tokens = self.vocabSize,
                                                                    output_mode = 'int',
                                                                    output_sequence_length = seqLen,
                                                                    standardize = self.standardation)
        self.textVectorization.adapt((self.encoderInput + ' [End]').to_numpy())
        self.encoderInput = self.textVectorization(self.encoderInput)
        self.encoderOutput = self.textVectorization(self.encoderOutput)
    def trainTestSplit(self):
        self.Data = [self.img,self.encoderInput,self.encoderOutput]
        random.shuffle(self.Data)
        numValSamples = int(self.validationSplit * len(self.Data[0]))
        numTrainSamples = int(self.trainSplit*len(self.Data[0]))
        trainData = [self.img[:numTrainSamples],self.encoderInput[:numTrainSamples],self.encoderOutput[:numTrainSamples]]
        valData = [self.img[numTrainSamples:numValSamples + numTrainSamples],
                    self.encoderInput[numTrainSamples:numValSamples + numTrainSamples],
                    self.encoderOutput[numTrainSamples:numValSamples + numTrainSamples]]
        testData = [self.img[numValSamples + numTrainSamples:],
                    self.encoderInput[numValSamples + numTrainSamples:],
                    self.encoderOutput[numValSamples + numTrainSamples:]]
        return [trainData,valData,testData]
    @staticmethod
    def standardation(inputString):
        punc = string.punctuation
        punc = punc.replace('[','')
        punc = punc.replace(']','')
        inputString = tf.strings.lower(inputString)
        inputString = tf.strings.regex_replace(inputString,f'[{punc}]','')
        return inputString