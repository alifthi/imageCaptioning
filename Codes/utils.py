import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import string

class Utils():
    def __init__(self) -> None:
        pass
    def readCSV(self):
        man = pd.read_csv(r"C:\Users\Legion\Documents\AI\Git\Image captioning\Data\images\image_captions.csv")
        self.imgName = man['img_name']
        self.imgCaption = '[Start] ' + man['img_caption'] + ' [End]'
    def loadImages(self):
        img = tf.keras.utils.image_dataset_from_directory("C:\\Users\\Legion\\Documents\\AI\\Git\\Image captioning\\Data\\")
        self.img = img.map(lambda x,y: x/255.0)
    def corpusPreprocessing(self):
        seqLen = 1024
        self.textVectorization = tf.keras.layers.TextVectorization(max_tokens = 200,
                                                                    output_mode = 'int',
                                                                    output_sequence_length = seqLen,
                                                                    standardize = self.standardation)
        self.textVectorization.adapt(self.imgCaption.to_numpy())
        self.vectorizedCaptions = self.textVectorization(self.imgCaption)
    @staticmethod
    def standardation(inputString):
        punc = string.punctuation
        punc = punc.replace('[','')
        punc = punc.replace(']','')
        inputString = tf.strings.lower(inputString)
        inputString = tf.strings.regex_replace(inputString,f'[{punc}]','')
        return inputString
util = Utils()
util.readCSV()
util.loadImages()
util.corpusPreprocessing()
