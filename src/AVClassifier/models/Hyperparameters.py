from .. import variables as v

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