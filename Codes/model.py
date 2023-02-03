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
        vggInput = ksl.Input(shape = [256,256,3])
        vgg = tf.keras.applications.vgg16.VGG16(include_top = False,
                                                weights = None,
                                                input_shape = [256,256,3])
        for l in vgg.layers:
            l.trainable = False
        vgg.load_weights(r"C:\Users\Legion\Downloads\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        vgg = vgg(vggInput)
        y = ksl.Flatten()(vgg)
        y = ksl.Dense(128,activation = 'relu')(y)
        y = ksl.Dense(1000,activation = 'relu')(y)
        y = ksl.Reshape([10,100])(y)
        captionsInput = ksl.Input([10])
        x = ksl.Embedding(input_dim = self.vocabSize,
                            output_dim = self.embeddingDim,
                            mask_zero = True)(captionsInput)
        x = ksl.concatenate([y,x],axis=-1)
        x = ksl.Bidirectional(ksl.LSTM(self.latentDim,dropout = 0.2,recurrent_dropout = 0.25),merge_mode = 'sum')(x)        
        x = ksl.Dense(self.vocabSize,activation = 'softmax')(x)          
        net = tf.keras.Model(inputs = [vggInput,captionsInput],outputs = x)
        return net
    def trainModel(self,trainData):
        self.Hist = self.net.fit(trainData[:-1],trainData[-1],epochs = 5,batch_size = 64)
    def compileModel(self):
        opt = optim.SGD(lr=0.1)  # we can use lr schedual
        self.net.compile(optimizer = opt,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
        self.net.summary()
    def plotHistory(slef):
        pass