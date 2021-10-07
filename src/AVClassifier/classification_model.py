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

    def evaluate(self, X, Y):
        if self.trainParameters.useGPU:
            with tf.device('/gpu:0'):
                return model.evaluate(X, Y, batch_size=self.hyperparameters.batchSize)
        return model.evaluate(X, Y, batch_size=self.hyperparameters.batchSize)
    
    def evaluateGenerator(self, dataGenerator):
        if self.trainParameters.useGPU:
            with tf.device('/gpu:0'):
                return model.evaluate(dataGenerator, batch_size=self.hyperparameters.batchSize)
        return model.evaluate(dataGenerator, batch_size=self.hyperparameters.batchSize)

    def test(self, X, Y, saveResults=False):
        # Generate predictions (probabilities)
        predictions = []
        if self.trainParameters.useGPU:
            with tf.device('/gpu:0'):
                for i in range(X_test.shape[0]):
                    predictions.append(model.predict(np.expand_dims(X[i], axis=0)))
        else:
            for i in range(X_test.shape[0]):
                    predictions.append(model.predict(np.expand_dims(X[i], axis=0)))

        predictions = np.concatenate(np.asarray(predictions))
        
        predicted = np.argmax(predictions, axis=1)
        real = np.argmax(Y, axis=1)

        #print('Predicted labels', predicted)
        #print('Actual    labels', real)

        confusionMatrix = confusion_matrix(real, predicted)
        #print('Confusion matrix: ', confusionMatrix)
        plt.figure()
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(pd.DataFrame(confusionMatrix, range(2), range(2)), annot=True, annot_kws={"size": 16}, cmap="YlGnBu") # font size
        plt.savefig(model_folder + model_name + '_confusion_matrix_heatmap.png')
        plt.show()

        TN, FP, FN, TP = confusionMatrix.ravel()

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)

        # Specificity or true negative rate
        TNR = TN/(TN+FP)

        # Precision or positive predictive value
        PPV = TP/(TP+FP)

        # Negative predictive value
        NPV = TN/(TN+FN)

        # Fall out or false positive rate
        FPR = FP/(FP+TN)

        # False negative rate
        FNR = FN/(TP+FN)

        # False discovery rate
        FDR = FP/(TP+FP)

        # Overall accuracy for each class
        ACC = (TP+TN)/(TP+FP+FN+TN)

        # F1-score
        F1Score = 2*TP/(2*TP+FP+FN)

        # print('Classes', classes)
        # print('Acuracy :', ACC)
        # print('Sensitivity :', TPR)
        # print('Specificity :', TNR)
        # print('Precision :', PPV)
        # print('F1-score :', F1Score)
        # print('FP: ', FP)
        # print('FN: ', FN)
        # print('TP: ', TP)
        # print('TN: ', TN)
        # print('\n\n')

        report = classification_report(real, predicted)
        with open(model_folder + model_name + '_classification_report.txt','w') as repfile:
          repfile.write(report)
        print(report)


        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(Y_test[:, 0], predictions[:, 0])
        #roc_auc = auc(fpr, tpr)
        roc_auc = roc_auc_score(predicted, real)

        # Plot ROC curve
        plt.style.use('seaborn')
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})')


        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend(loc="lower right")
        plt.savefig(model_folder + model_name + '_roc.png')
        plt.show()

        
  
    def testGenerator(self, dataGenerator, saveResults=False):
        pass
    
    def saveModelWeights(self, path=None):
        self.model.save_weights(self.weightsFile if path is None else path)

    def loadModelWeights(self, path=None):
        self.model.load_weights(self.weightsFile if path is None else path)

    def printModelSummary(self):
        self.model.summary()
