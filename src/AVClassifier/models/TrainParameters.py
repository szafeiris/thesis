from .. import variables as v

class TrainParameters:
    def __init__(self, useGPU=v.USE_GPU, verbosity=v.VERBOSE, trainShuffle=v.TRAIN_SHUFFLE,
                 useMp=v.USE_MP, workers=v.WORKERS, saveFigures=v.SAVE_FIGURES):
        self.useGPU = useGPU
        self.verbosity = verbosity
        self.trainShuffle = trainShuffle
        self.useMp = useMp
        self.workers = workers
        self.saveFigures = saveFigures