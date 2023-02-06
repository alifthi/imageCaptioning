from utils import Utils
from model import Model
util = Utils()
util.readCSV()
util.loadImages()
util.corpusPreprocessing()
trainData,valData,testData = util.trainTestSplit()
model = Model()
model.net = model.buildModel()
model.compileModel()

model.trainModel(trainData,validation = valData)