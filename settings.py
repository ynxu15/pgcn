# configuration of our models

class Config():
    def __init__(self, modelName):
        # default settings of model
        self.modelName = modelName
        self.embedSize, self.lr = 64, 0.001
        self.topN = 50            # the maximum number of neighbors
        self.mode = 2             # 0: traditional MF, 1: use only user-item graph, 2: use all graphs
        self.convLayerNum = 2     # Number of convolutional layers:  1, 2, 3
        self.dis = 2              # the largest distance between neighbors: 1, 2
        self.regValue = 5e-5      # regularization weight, from 1e-5 to 1e-4

        if modelName == 'pgcna':
            # attention
            self.attMode = 0      # 0: element wise product, 1: concat
            self.attActive = 0    # 0: relu, 1: sigmoid, 2: tanh
            self.beta = 0.5
