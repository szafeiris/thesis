from . import variables as v
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Input, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc


class Hyperparameters:
    def __init__( self, epochs=v.EPOCHS, batchSize=v.BATCH_SIZE, 
                  useBatchNorm=v.USE_BATCH_NORM, useDropout=v.USE_DROPOUT, 
                  baseFilters=v.BASE_FILTERS, learningRate=v.LEARNING_RATE, 
                  decay=v.DECAY, lossFun=v.LOSS_FUN, dropoutProbs=v.DROPOUT_PROBS):
        self.epochs = epochs
        self.batchSize = batchSize 
        self.useBatchNorm = useBatchNorm 
        self.useDropout = useDropout 
        self.baseFilters = baseFilters
        self.learningRate = learningRate
        self.decay = decay
        self.lossFun = lossFun
        self.dropoutProbs = dropoutProbs

class TrainParameters:
    def __init__(self, useGPU=v.USE_GPU, verbosity=v.VERBOSE, trainShuffle=v.TRAIN_SHUFFLE,
                 useMp=v.USE_MP, workers=v.WORKERS):
        self.useGPU = useGPU
        self.verbosity = verbosity
        self.trainShuffle = trainShuffle
        self.useMp = useMp
        self.workers = workers

        

class Model3D:
    def __init__( self, modelName, nClasses=2, dataShape=v.INPUT_DATA_SHAPE_3D, 
                  hyperparameters=Hyperparameters(), trainParameters=TrainParameters()):
        self.modelName = modelName
        self.nClases = nClasses
        self.dataShape = dataShape
        self.hyperparameters = hyperparameters
        self.trainParameters = trainParameters
        
        # Create paths if they not exist
        self.modelPath = f"{v.MODELS_DATA_PATH}{modelName}/"
        if not os.path.exists(self.modelPath):
            os.mkdir(self.modelPath)

        self.weightsPath = f"{self.modelPath}/weights/"
        self.weightsFile = f"{self.weightsPath}{self.modelName}_w.h5"
        if not os.path.exists(self.weightsPath):
            os.mkdir(self.weightsPath)

        self.modelCheckpoint = f"{self.modelPath}checkpoint.h5"
        self.modelGraphs = f"{self.modelPath}/graphs/"
        if not os.path.exists(self.modelGraphs):
            os.mkdir(self.modelGraphs)

        # Build the model
        self.model = Sequential()
        self.model.add(Input(shape=self.dataShape, name='input_1'))

        self.model.add(Conv3D(self.hyperparameters.baseFilters, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block1_conv1'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block1_conv2'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), name='block1_pool'))
        
        if self.hyperparameters.useBatchNorm:
            self.model.add(BatchNormalization(axis=-1))
        
        self.model.add(Conv3D(self.hyperparameters.baseFilters*2, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block2_conv1'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*2, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block2_conv2'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), name='block2_pool'))

        if self.hyperparameters.useBatchNorm:
            self.model.add(BatchNormalization(axis=-1))

        self.model.add(Conv3D(self.hyperparameters.baseFilters*4, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block3_conv1'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*4, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block3_conv2'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*4, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block3_conv3'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), name='block3_pool'))
        
        if self.hyperparameters.useBatchNorm:
          self.model.add(BatchNormalization(axis=-1))
        
        self.model.add(Conv3D(self.hyperparameters.baseFilters*8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block4_conv1'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block4_conv2'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block4_conv3'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), name='block4_pool'))
        
        if self.hyperparameters.useBatchNorm:
          self.model.add(BatchNormalization(axis=-1))
        
        self.model.add(Conv3D(self.hyperparameters.baseFilters*8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block5_conv1'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block5_conv2'))
        self.model.add(Conv3D(self.hyperparameters.baseFilters*8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='block5_conv3'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), name='block5_pool'))
        
        if self.hyperparameters.useBatchNorm:
          self.model.add(BatchNormalization(axis=-1))
        
        self.model.add(Flatten(name='flatten_1'))
        self.model.add(Dense(2048, activation='relu', name='dense_1', kernel_initializer='he_uniform'))

        if self.hyperparameters.useDropout:
            self.model.add(Dropout(self.hyperparameters.dropoutProbs[0]))

        self.model.add(Dense(2048, activation='relu', name='dense_0', kernel_initializer='he_uniform'))
        
        if self.hyperparameters.useDropout:
            self.model.add(Dropout(self.hyperparameters.dropoutProbs[1]))
    
        self.model.add(Dense(nClasses, name='output_1', activation='softmax'))
        
        self.model.compile( adam_v2.Adam(learning_rate=self.hyperparameters.learningRate, decay=self.hyperparameters.decay),
                            loss=self.hyperparameters.lossFun, metrics=["accuracy"])
    
    def __trainModel(self, X, Y, validationData):
        trainCallbacks = [EarlyStopping(monitor='loss', patience=10)]

        if self.trainParameters.useCheckpoint:
            trainCallbacks.append(ModelCheckpoint( self.modelCheckpoint, 
                                                   monitor='loss', 
                                                   mode='min', 
                                                   verbose=1, 
                                                   save_best_only=True, 
                                                   save_weights_only=True))

        self.history = self.model.fit( x=X, y=Y,
                                       batch_size=self.hyperparameters.batchSize,
                                       epochs=self.hyperparameters.epochs,
                                       verbose=self.trainParameters.verbosity, 
                                       shuffle=self.trainParameters.trainShuffle, 
                                       callbacks=trainCallbacks, 
                                       validation_data=validationData,
                                       use_multiprocessing=self.trainParameters.useMp, 
                                       workers=self.trainParameters.workers
                                     )
        return self.history

    def __trainGenModel(self, dataGenerator, validationData):
        trainCallbacks = [EarlyStopping(monitor='loss', patience=10)]

        if self.trainParameters.useCheckpoint:
            trainCallbacks.append(ModelCheckpoint( self.modelCheckpoint, 
                                                   monitor='loss', 
                                                   mode='min', 
                                                   verbose=1, 
                                                   save_best_only=True, 
                                                   save_weights_only=True))

        self.history = self.model.fit( x=dataGenerator,
                                       batch_size=self.hyperparameters.batchSize,
                                       epochs=self.hyperparameters.epochs,
                                       verbose=self.trainParameters.verbosity, 
                                       shuffle=self.trainParameters.trainShuffle, 
                                       callbacks=trainCallbacks, 
                                       validation_data=validationData,
                                       use_multiprocessing=self.trainParameters.useMp, 
                                       workers=self.trainParameters.workers
                                     )
        return self.history

    def train(self, X, Y, validationData=None):
        if self.trainParameters.useGPU:
            with tf.device('/device:GPU:0'):
                hist = self.__trainModel(X, Y, validationData)
        else:
            hist = self.__trainModel(X, Y, validationData)

        return hist
    
    def trainGenerator(self, dataGenerator, validationData=None):
        if self.trainParameters.useGPU:
            with tf.device('/device:GPU:0'):
                hist = self.__trainGenModel(dataGenerator, validationData)
        else:
            hist = self.__trainGenModel(dataGenerator, validationData)

        return hist
    
    def saveModelWeights(self, path=None):
        self.model.save_weights(self.weightsFile if path is None else path)

    def loadModelWeights(self, path=None):
        self.model.load_weights(self.weightsFile if path is None else path)

    def printModelSummary(self):
        self.model.summary()
