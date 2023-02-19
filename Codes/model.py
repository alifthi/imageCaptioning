import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers as ksl
import tensorflow as tf
from tensorflow.keras import optimizers as optim
class Model():
    def __init__(self,vocabSize = 2000) -> None:
        self.embeddingDim = 1024
        self.vocabSize = vocabSize
        self.latentDim = 256
    def buildModel(self):
        imageInputs = ksl.Input(shape = [256,256,3])
        x = ksl.Conv2D(16,kernel_size = 3,strides = 2,activation = 'relu',padding = 'same')(imageInputs)
        x = ksl.MaxPooling2D(2,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)   
        
        x = ksl.Conv2D(8,kernel_size = 3,strides = 2,activation = 'relu',padding = 'same')(x)
        x = ksl.MaxPooling2D(2,padding = 'same')(x)
        x = ksl.BatchNormalization()(x)

        y = ksl.Flatten()(x)
        y = ksl.Dense(128,activation = 'relu')(y)
        y = ksl.Dense(self.latentDim,activation = 'relu')(y)
        captionsInput = ksl.Input([10])
        x = ksl.Embedding(input_dim = self.vocabSize,
                            output_dim = self.embeddingDim,
                            mask_zero = True)(captionsInput)
        x = ksl.GRU(self.latentDim,dropout = 0.2,recurrent_dropout = 0.25, return_sequences=True)(x,initial_state = y)        
        x = ksl.TimeDistributed(ksl.Dropout(0.5))(x)
        x = ksl.TimeDistributed(ksl.Dense(self.vocabSize, activation="softmax"))(x)      
        net = tf.keras.Model(inputs = [imageInputs,captionsInput],outputs = x)
        return net
    def trainModel(self,trainData,validation):
        self.Hist = self.net.fit(trainData[:-1],trainData[-1],epochs = 150,batch_size = 64,validation_data = [[validation[:-1]],validation[-1]])
    def compileModel(self):
        opt = optim.SGD(lr=0.1)  
        self.net.compile(optimizer = opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(),metrics = ['accuracy'])
        self.net.summary()
    @staticmethod
    def plotHistory(Hist):
        from matplotlib import pyplot as plt
        plt.plot(Hist.history['accuracy'])
        plt.plot(Hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.plot(Hist.history['loss'])
        plt.plot(Hist.history['val_loss'])
        plt.title('model loss')