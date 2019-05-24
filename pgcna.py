#Authors: Yanan Xu<ynxu15@gmail.com>
#License: GNU General Public License v2.0

import numpy as np
import tensorflow as tf

class PGCN_A(object):
    """Class for Path conditioned Graph Convolutional Network
    ------------
    Learning Shared Vertex Representation in Heterogeneous Graphs with
    Convolutional Networks for Recommendation. Yanan Xu, 2019.
    """
    def __init__(self, userNum, itemNum, gramNum, config, initialize = True,
                 userItemDic=None, itemUserDic=None, userGramSeq1Dic=None, itemItemSeq1Dic=None,
                 userUserIntDic=None, itemItemIntDic=None, userUserSeq2Dic=None, itemItemSeq2Dic=None
                 ):
        self.userNum, self.itemNum, self.gramNum = userNum, itemNum, gramNum
        self.embedSize, self.weightSize, self.lr = config.embedSize, config.embedSize, config.lr
        self.initialize = initialize
        self.topN = config.topN                     # the maximum number of neighbors for each vertex
        self.mode = config.mode                     # 0: traditional MF, 1: use only user-item graph, 2: use all graphs
        self.convLayerNum = config.convLayerNum     # Number of convolutional layers:  1, 2, 3
        self.dis = config.dis                       # the largest distance between neighbors: 1, 2
        self.regValue = config.regValue             # regularization weight

        # attention
        self.attMode = config.attMode               # 0: element wise product, 1: concat
        self.attActive = config.attActive           # 0: relu, 1: sigmoid, 2: tanh
        self.beta = config.beta

        # neighbors with distance = 1
        self.userItemDic, self.itemUserDic = userItemDic, itemUserDic
        self.userGramSeq1Dic, self.itemItemSeq1Dic = userGramSeq1Dic, itemItemSeq1Dic

        # neighbors with distance = 2
        self.userUserIntDic, self.itemItemIntDic = userUserIntDic, itemItemIntDic
        self.userUserSeq2Dic, self.itemItemSeq2Dic = userUserSeq2Dic, itemItemSeq2Dic

        if self.initialize:
            self.initialize_graph()

    def printSettings(self):
        # print settings
        print('*'*80)
        print('pgcna model:')
        print('topN: %d, embedSize: %d, mode: %d, convLayers: %d, dis: %d, regValue: %.6f'%(self.topN, self.embedSize, self.mode, self.convLayerNum, self.dis, self.regValue))
        print('attMode: %d, attActive: %d, beta: %.4f'%(self.attMode, self.attActive, self.beta))
        print('*'*80)

    def get_graph_data(self):
        userNum, itemNum, gramNum = self.userNum, self.itemNum, self.gramNum
        embedSize, topN = self.embedSize, self.topN
        if self.dis>0 and self.mode>0:
            # user-item graph, neighbors' distance = 1,
            userItemMatrix = np.zeros((userNum, topN), dtype=int)
            itemUserMatrix = np.zeros((itemNum, topN), dtype=int)
            userItemMatrix[:, :], itemUserMatrix[:, :] = itemNum, userNum

            for u in self.userItemDic:
                ddic = self.userItemDic[u]
                neighbor = list(ddic.keys())
                userItemMatrix[u, :len(neighbor)] = neighbor
            for i in self.itemUserDic:
                ddic = self.itemUserDic[i]
                neighbor = list(ddic.keys())
                itemUserMatrix[i, :len(neighbor)] = neighbor

            self.userItemMatrix = tf.constant(userItemMatrix, dtype=tf.int32,shape=(userNum, topN),name='userItemMatrix')
            self.itemUserMatrix = tf.constant(itemUserMatrix, dtype=tf.int32,shape=(itemNum, topN),name='itemUserMatrix')

            del userItemMatrix, itemUserMatrix
            self.userItemDic, self.itemUserDic = [], []

        if self.dis>0 and self.mode >1:
            # user-gram graph, item-item graph, neighbors distance = 1,
            userGramMatrix = np.zeros((userNum, topN), dtype=int)
            itemItemMatrix = np.zeros((itemNum, topN), dtype=int)
            userGramMatrix[:, :], itemItemMatrix[:, :] = gramNum, itemNum

            for u in self.userGramSeq1Dic:
                ddic = self.userGramSeq1Dic[u]
                neighbor = list(ddic.keys())
                userGramMatrix[u, :len(neighbor)] = neighbor
            for i in self.itemItemSeq1Dic:
                ddic = self.itemItemSeq1Dic[i]
                neighbor = list(ddic.keys())
                itemItemMatrix[i, :len(neighbor)] = neighbor

            self.userGramMatrix = tf.constant(userGramMatrix, dtype=tf.int32,shape=(userNum, topN),name='userItemMatrix')
            self.itemItemMatrix = tf.constant(itemItemMatrix, dtype=tf.int32,shape=(itemNum, topN),name='itemUserMatrix')

            del userGramMatrix, itemItemMatrix
            self.userGramSeq1Dic, self.itemItemSeq1Dic  = [], []

        if self.dis > 1 and self.mode > 0:
            # user-item graph, neighbors' distance = 2,
            userUserIntMatrix = np.zeros((userNum, topN), dtype=int)
            itemItemIntMatrix = np.zeros((itemNum, topN), dtype=int)
            userUserIntMatrix[:, :], itemItemIntMatrix[:, :] = userNum,itemNum

            for u in self.userUserIntDic:
                ddic = self.userUserIntDic[u]
                neighbor = list(ddic.keys())
                userUserIntMatrix[u, :len(neighbor)] = neighbor
            for i in self.itemItemIntDic:
                ddic = self.itemItemIntDic[i]
                neighbor = list(ddic.keys())
                itemItemIntMatrix[i, :len(neighbor)] = neighbor

            self.userUserIntMatrix = tf.constant(userUserIntMatrix, dtype=tf.int32, shape=(userNum, topN), name='userUserIntMatrix')
            self.itemItemIntMatrix = tf.constant(itemItemIntMatrix, dtype=tf.int32, shape=(itemNum, topN), name='itemItemIntMatrix')

            del userUserIntMatrix, itemItemIntMatrix
            self.userUserIntDic, self.itemItemIntDic = [], []

        if self.dis>1 and self.mode>1:
            # user-gram graph, item-item graph, neighbors' distance = 2,
            userUserSeq2Matrix = np.zeros((userNum, topN), dtype=int)
            itemItemSeq2Matrix = np.zeros((itemNum, topN), dtype=int)
            userUserSeq2Matrix[:, :], itemItemSeq2Matrix[:, :] = userNum, itemNum

            for u in self.userUserSeq2Dic:
                ddic = self.userUserSeq2Dic[u]
                neighbor = list(ddic.keys())
                userUserSeq2Matrix[u, :len(neighbor)] = neighbor
            for i in self.itemItemSeq2Dic:
                ddic = self.itemItemSeq2Dic[i]
                neighbor = list(ddic.keys())
                itemItemSeq2Matrix[i, :len(neighbor)] = neighbor

            self.userUserSeq2Matrix = tf.constant(userUserSeq2Matrix, dtype=tf.int32,shape=(userNum, topN),name='userUserSeq2Matrix')
            self.itemItemSeq2Matrix = tf.constant(itemItemSeq2Matrix, dtype=tf.int32,shape=(itemNum, topN),name='itemItemSeq2Matrix')

            del userUserSeq2Matrix, itemItemSeq2Matrix
            self.userUserSeq2Dic, self.itemItemSeq2Dic  = [], []

    def initialize_graph(self):
        self._setup_base_graph()                              # setup the graph
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.init_op = tf.global_variables_initializer()
            self.sess.run(self.init_op)
        self.initialized = True

    def _setup_base_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.get_graph_data()
            self.item = tf.placeholder(dtype=tf.int32, shape=[None], name="item")
            self.user = tf.placeholder(dtype=tf.int32, shape=[None], name="user")

            self.r = tf.placeholder(dtype=tf.float32, shape=[None], name="rating")
            self._setup_variables()
            self._setup_training()

    def _setup_variables(self):
        userNum, itemNum, gramNum = self.userNum, self.itemNum, self.gramNum
        embedSize, topN = self.embedSize, self.topN
        _getVar = self._getVar

        # Embeddings for user and items
        self.uEmbed = self._getEmbed((userNum, embedSize), 'userEmbedding')
        self.iEmbed = self._getEmbed((itemNum, embedSize), 'itemEmbedding')

        # attention weight
        inWsize, hidWsize = embedSize, self.weightSize

        if self.attMode == 1:
            inWsize = inWsize * 2

        if self.mode > 0:
            self.itemMask = tf.constant(value=[1] * itemNum + [0], dtype=tf.float32, name='itemMask')
            self.userMask = tf.constant(value=[1] * userNum + [0], dtype=tf.float32, name='userMask')
            self.attWUserItem, self.attWItemUser = [], []
            self.attBUserItem, self.attBItemUser = [], []
            self.attWUserItem1,self.attWItemUser1 = [], []
            self.userW,self.itemW = [], []
            self.userItemNW, self.itemUserNW = [], []

            for i in range(self.convLayerNum):
                # attention weight
                self.attWUserItem.append(_getVar((inWsize, hidWsize), 'attWUserItem%d' % (i)))
                self.attWItemUser.append(_getVar((inWsize, hidWsize), 'attWItemUser%d' % (i)))

                self.attBUserItem.append(_getVar((1, hidWsize), 'attBUserItem%d'%(i)))
                self.attBItemUser.append( _getVar((1, hidWsize), 'attBItemUser%d'%(i)))

                self.attWUserItem1.append(_getVar((hidWsize,1), 'attWUserItem1%d'%(i)))
                self.attWItemUser1.append(_getVar((hidWsize,1), 'attWItemUser1%d'%(i)))

                # convolution weight
                self.userW.append(_getVar((embedSize, embedSize), 'userW%d'%(i)))
                self.itemW.append(_getVar((embedSize, embedSize), 'itemW%d'%(i)))
                self.userItemNW.append(_getVar((embedSize, embedSize), 'userItemNW%d'%(i)))
                self.itemUserNW.append(_getVar((embedSize, embedSize), 'itemUserNW%d'%(i)))

            if self.dis>1:
                self.attWUserUserInt, self.attWItemItemInt = [], []
                self.attBUserUserInt, self.attBItemItemInt = [], []
                self.attWUserUserInt1, self.attWItemItemInt1 = [], []
                self.userUserIntNW, self.itemItemIntNW = [], []

                for i in range(self.convLayerNum):
                    # attention weight
                    self.attWUserUserInt.append(_getVar((inWsize, hidWsize), 'attWUserUserInt%d'%(i)))
                    self.attWItemItemInt.append(_getVar((inWsize, hidWsize), 'attWItemItemInt%d'%(i)))

                    self.attBUserUserInt.append(_getVar((1, hidWsize), 'attBUserUserInt%d'%(i)))
                    self.attBItemItemInt.append(_getVar((1, hidWsize), 'attBItemItemInt%d'%(i)))

                    self.attWUserUserInt1.append(_getVar((hidWsize, 1), 'attWUserUserInt1%d'%(i)))
                    self.attWItemItemInt1.append(_getVar((hidWsize, 1), 'attWItemItemInt1%d'%(i)))

                    # convolution weight
                    self.userUserIntNW.append(self._getVar((embedSize, embedSize), 'userUserIntNW%d' % (i)))
                    self.itemItemIntNW.append(self._getVar((embedSize, embedSize), 'itemItemIntNW%d' % (i)))
        if self.mode >1:
            self.gEmbed = self._getEmbed((gramNum, embedSize), 'gramEmbedding')
            self.gramMask = tf.constant(value=[1] * gramNum + [0], dtype=tf.float32, name='gramMask')

            self.attWUserGram, self.attWItemItem = [], []
            self.attBUserGram, self.attBItemItem = [], []
            self.attWUserGram1, self.attWItemItem1 = [], []
            self.attWUserGram, self.userGramNW, self.itemItemNW = [], [], []
            for i in range(self.convLayerNum):
                # attention weight
                self.attWUserGram.append(_getVar((inWsize, hidWsize), 'attWUserGram%d' % (i)))
                self.attWItemItem.append(_getVar((inWsize, hidWsize), 'attWItemItem%d' % (i)))

                self.attBUserGram.append(_getVar((1, hidWsize), 'attBUserGram%d' % (i)))
                self.attBItemItem.append(_getVar((1, hidWsize), 'attBItemItem%d' % (i)))

                self.attWUserGram1.append(_getVar((hidWsize, 1), 'attWUserGram1%d' % (i)))
                self.attWItemItem1.append(_getVar((hidWsize, 1), 'attWItemItem1%d' % (i)))

                # convolution weight
                self.userGramNW.append(_getVar((embedSize, embedSize), 'userGramNW%d'%(i)))
                self.itemItemNW.append(_getVar((embedSize, embedSize), 'itemItemNW%d'%(i)))

            if self.dis>1:
                self.attWUserUserSeq2, self.attWItemItemSeq2 = [], []
                self.attBUserUserSeq2, self.attBItemItemSeq2 = [], []
                self.attWUserUserSeq21, self.attWItemItemSeq21 = [], []
                self.userUserSeq2NW, self.itemItemSeq2NW = [], []
                for i in range(self.convLayerNum):
                    # attention weight
                    self.attWUserUserSeq2.append(_getVar((inWsize, hidWsize), 'attWUserUserSeq2%d' % (i)))
                    self.attWItemItemSeq2.append(_getVar((inWsize, hidWsize), 'attWItemItemSeq2%d' % (i)))

                    self.attBUserUserSeq2.append(_getVar((1, hidWsize), 'attBUserUserSeq2%d' % (i)))
                    self.attBItemItemSeq2.append(_getVar((1, hidWsize), 'attBItemItemSeq2%d' % (i)))

                    self.attWUserUserSeq21.append(_getVar((hidWsize, 1), 'attWUserUserSeq21%d' % (i)))
                    self.attWItemItemSeq21.append(_getVar((hidWsize, 1), 'attWItemItemSeq21%d' % (i)))

                    # convolution weight
                    self.userUserSeq2NW.append(_getVar((embedSize, embedSize), 'userUserSeq2NW%d' % (i)))
                    self.itemItemSeq2NW.append(_getVar((embedSize, embedSize), 'itemItemSeq2NW%d' % (i)))

        self.saver = tf.train.Saver(max_to_keep=10)


    def _attMLP(self, q_, mask_mat, W, b, h):
       '''MLP version attention'''
       with tf.name_scope("attention_MLP"):
            r = (self.attMode+1)* self.embedSize                   # 0: element wise 1: concat

            MLP_output = tf.matmul(tf.reshape(q_,[-1,r]), W) + b   #(B*N, E or 2*E) * (E or 2*E, w) + (1, w)
            if self.attActive == 0:
                MLP_output = tf.nn.relu(MLP_output)
            elif self.attActive == 1:
                MLP_output = tf.nn.sigmoid(MLP_output)
            elif self.attActive == 2:
                MLP_output = tf.nn.tanh( MLP_output )

            A_ = tf.reshape(tf.matmul(MLP_output, h),[-1,self.topN, 1]) #(B*N, w) * (w, 1) => (None, 1) => (B, N)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            mask_mat = tf.reshape(mask_mat, (-1, self.topN, 1))
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True) +0.00001  # (B, 1, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))
            A = tf.div(exp_A_, exp_sum)  # (B, N, 1)

            return A, q_

    def _setup_training(self):
        embedSize, topN = self.embedSize, self.topN
        uEmbedTmp, iEmbedTmp = self.uEmbed, self.iEmbed
        _attMLP = self._attMLP

        if self.mode == 0:
            pass

        if self.mode == 1:
            for i in range(self.convLayerNum):
                # neighbors' distance = 1
                userItemNE = tf.gather(iEmbedTmp, self.userItemMatrix)  # userNum*N*E
                itemUserNE = tf.gather(uEmbedTmp, self.itemUserMatrix)  # itemNum*N*E

                userItemNM = tf.gather(self.itemMask, self.userItemMatrix)  # userNum*N*1
                itemUserNM = tf.gather(self.userMask, self.itemUserMatrix)  # itemNum*N*1

                uE1 = tf.reshape(uEmbedTmp, (-1, 1, embedSize))  # userNum*1*E
                iE1 = tf.reshape(iEmbedTmp, (-1, 1, embedSize))  # itemNum*1*E
                uE1, iE1 = uE1[:-1, :, :], iE1[:-1, :, :]

                if self.attMode == 0:
                    userItemW1, esum1 = _attMLP(uE1 * userItemNE, userItemNM, self.attWUserItem[i],
                                                    self.attBUserItem[i], self.attWUserItem1[i])
                    itemUserW1, esum = _attMLP(iE1 * itemUserNE, itemUserNM, self.attWItemUser[i],
                                                   self.attBItemUser[i], self.attWItemUser1[i])
                else:
                    userItemW1, esum1 = _attMLP(tf.concat([userItemNE, tf.tile(uE1, tf.stack([1, topN, 1]))], 2),
                                                    userItemNM, self.attWUserItem[i], self.attBUserItem[i], self.attWUserItem1[i])
                    itemUserW1, esum = _attMLP(tf.concat([itemUserNE, tf.tile(iE1, tf.stack([1, topN, 1]))], 2),
                                                   itemUserNM, self.attWItemUser[i], self.attBItemUser[i], self.attWItemUser1[i])

                userItemNESum = tf.reduce_sum(userItemNE * userItemW1, axis=1)  # userNum*E
                itemUserNESum = tf.reduce_sum(itemUserNE * itemUserW1, axis=1)  # ItemNum*E

                zero_vector_u = tf.constant([0.0]*embedSize, tf.float32, [1, embedSize])
                userItemNESum = tf.concat([userItemNESum, zero_vector_u], 0)
                zero_vector_i = tf.constant([0.0]*embedSize, tf.float32, [1, embedSize])
                itemUserNESum = tf.concat([itemUserNESum, zero_vector_i], 0)

                if self.dis == 1:
                    uEmbedTmp = tf.matmul(uEmbedTmp, self.userW[i]) + tf.matmul(userItemNESum, self.userItemNW[i])
                    iEmbedTmp = tf.matmul(iEmbedTmp, self.itemW[i]) + tf.matmul(itemUserNESum, self.itemUserNW[i])

                if self.dis == 2:
                    # neighbors' distance = 2
                    userUserIntNE = tf.gather(uEmbedTmp, self.userUserIntMatrix)  # userNum*N*E
                    itemItemIntNE = tf.gather(iEmbedTmp, self.itemItemIntMatrix)  # itemNum*N*E

                    userUserIntNM = tf.gather(self.userMask, self.userUserIntMatrix)  # userNum*N*1
                    itemItemIntNM = tf.gather(self.itemMask, self.itemItemIntMatrix)  # itemNum*N*1

                    if self.attMode == 0:
                        userUserIntW1, esum1 = _attMLP(uE1 * userUserIntNE, userUserIntNM, self.attWUserUserInt[i],
                                                       self.attBUserUserInt[i], self.attWUserUserInt1[i])
                        itemItemIntW1, esum = _attMLP(iE1 * itemItemIntNE, itemItemIntNM, self.attWItemItemInt[i],
                                                      self.attBItemItemInt[i], self.attWItemItemInt1[i])
                    else:
                        userUserIntW1, esum1 = _attMLP(tf.concat([userUserIntNE, tf.tile(uE1, tf.stack([1, topN, 1]))], 2), userUserIntNM,
                                                       self.attWUserUserInt[i], self.attBUserUserInt[i], self.attWUserUserInt1[i])
                        itemItemIntW1, esum = _attMLP(tf.concat([itemItemIntNE, tf.tile(iE1, tf.stack([1, topN, 1]))], 2), itemItemIntNM,
                                                      self.attWItemItemInt[i], self.attBItemItemInt[i], self.attWItemItemInt1[i])

                    userUserIntNESum = tf.reduce_sum(userUserIntNE * userUserIntW1, axis=1)  # userNum*E
                    itemItemIntNESum = tf.reduce_sum(itemItemIntNE * itemItemIntW1, axis=1)  # ItemNum*E

                    zero_vector_u = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    userUserIntNESum = tf.concat([userUserIntNESum, zero_vector_u], 0)
                    zero_vector_i = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    itemItemIntNESum = tf.concat([itemItemIntNESum, zero_vector_i], 0)

                    uEmbedTmp = tf.matmul(uEmbedTmp, self.userW[i]) + tf.matmul(userItemNESum, self.userItemNW[i]) \
                                + tf.matmul(userUserIntNESum, self.userUserIntNW[i])
                    iEmbedTmp = tf.matmul(iEmbedTmp, self.itemW[i]) + tf.matmul(itemUserNESum, self.itemUserNW[i])\
                                + tf.matmul(itemItemIntNESum, self.itemItemIntNW[i])

        if self.mode == 2:
            gEmbedTmp = self.gEmbed
            for i in range(self.convLayerNum):
                userItemNE = tf.gather(iEmbedTmp, self.userItemMatrix)  # userNum*N*E
                itemUserNE = tf.gather(uEmbedTmp, self.itemUserMatrix)  # itemNum*N*E
                userGramNE = tf.gather(gEmbedTmp, self.userGramMatrix)  # userNum*N*E
                itemItemNE = tf.gather(iEmbedTmp, self.itemItemMatrix)  # itemNum*N*E

                userItemNM = tf.gather(self.itemMask, self.userItemMatrix)  # userNum*N*1
                itemUserNM = tf.gather(self.userMask, self.itemUserMatrix)  # itemNum*N*1
                userGramNM = tf.gather(self.gramMask, self.userGramMatrix)  # userNum*N*1
                itemItemNM = tf.gather(self.itemMask, self.itemItemMatrix)  # itemNum*N*1

                uE1 = tf.reshape(uEmbedTmp, (-1, 1, embedSize)) # userNum*1*E
                iE1 = tf.reshape(iEmbedTmp, (-1, 1, embedSize)) # itemNum*1*E
                uE1, iE1 = uE1[:-1, :, :], iE1[:-1, :, :]

                if self.attMode == 0:
                    userItemW1, esum1 = _attMLP(uE1 * userItemNE, userItemNM, self.attWUserItem[i],
                                                self.attBUserItem[i], self.attWUserItem1[i])
                    itemUserW1, esum = _attMLP(iE1 * itemUserNE, itemUserNM, self.attWItemUser[i],
                                               self.attBItemUser[i], self.attWItemUser1[i])
                    userGramW1, esum = _attMLP(uE1 * userGramNE, userGramNM, self.attWUserGram[i],
                                               self.attBUserGram[i], self.attWUserGram1[i])
                    itemItemW1, esum = _attMLP(iE1 * itemItemNE, itemItemNM, self.attWItemItem[i],
                                               self.attBItemItem[i], self.attWItemItem1[i])
                else:
                    userItemW1, esum1 = _attMLP(
                        tf.concat([userItemNE, tf.tile(uE1, tf.stack([1, topN, 1]))], 2), userItemNM,
                        self.attWUserItem[i], self.attBUserItem[i], self.attWUserItem1[i])
                    itemUserW1, esum = _attMLP(
                        tf.concat([itemUserNE, tf.tile(iE1, tf.stack([1, topN, 1]))], 2), itemUserNM,
                        self.attWItemUser[i], self.attBItemUser[i], self.attWItemUser1[i])
                    userGramW1, esum = _attMLP(
                        tf.concat([userGramNE, tf.tile(uE1, tf.stack([1, topN, 1]))], 2), userGramNM,
                        self.attWUserGram[i], self.attBUserGram[i], self.attWUserGram1[i])
                    itemItemW1, esum = _attMLP(
                        tf.concat([itemItemNE, tf.tile(iE1, tf.stack([1, topN, 1]))], 2), itemItemNM,
                        self.attWItemItem[i], self.attBItemItem[i], self.attWItemItem1[i])

                userItemNESum = tf.reduce_sum(userItemNE * userItemW1, axis=1)  # userNum*E
                itemUserNESum = tf.reduce_sum(itemUserNE * itemUserW1, axis=1)  # ItemNum*E

                userGramNESum = tf.reduce_sum(userGramNE * userGramW1, axis=1)  # userNum*E
                itemItemNESum = tf.reduce_sum(itemItemNE * itemItemW1, axis=1)  # itemNum*E

                zero_vector_u = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                userItemNESum = tf.concat([userItemNESum, zero_vector_u], 0)
                zero_vector_i = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                itemUserNESum = tf.concat([itemUserNESum, zero_vector_i], 0)

                zero_vector_u = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                userGramNESum = tf.concat([userGramNESum, zero_vector_u], 0)
                zero_vector_i = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                itemItemNESum = tf.concat([itemItemNESum, zero_vector_i], 0)

                if self.dis == 1:
                    uEmbedTmp = tf.matmul(uEmbedTmp, self.userW[i]) + tf.matmul(userItemNESum, self.userItemNW[i]) + tf.matmul(
                        userGramNESum, self.userGramNW[i])
                    iEmbedTmp = tf.matmul(iEmbedTmp, self.itemW[i]) + tf.matmul(itemUserNESum, self.itemUserNW[i]) + tf.matmul(
                        itemItemNESum, self.itemItemNW[i])

                if self.dis == 2:
                    # user-item graph, neighbors' distance = 2
                    userUserIntNE = tf.gather(uEmbedTmp, self.userUserIntMatrix)  # userNum*N*E
                    itemItemIntNE = tf.gather(iEmbedTmp, self.itemItemIntMatrix)  # itemNum*N*E

                    userUserIntNM = tf.gather(self.userMask, self.userUserIntMatrix)  # userNum*N*1
                    itemItemIntNM = tf.gather(self.itemMask, self.itemItemIntMatrix)  # itemNum*N*1

                    if self.attMode == 0:
                        userUserIntW1, esum1 = _attMLP(uE1 * userUserIntNE, userUserIntNM, self.attWUserUserInt[i],
                                                       self.attBUserUserInt[i], self.attWUserUserInt1[i])
                        itemItemIntW1, esum = _attMLP(iE1 * itemItemIntNE, itemItemIntNM, self.attWItemItemInt[i],
                                                      self.attBItemItemInt[i], self.attWItemItemInt1[i])
                    else:
                        userUserIntW1, esum1 = _attMLP(
                            tf.concat([userUserIntNE, tf.tile(uE1, tf.stack([1, topN, 1]))], 2), userUserIntNM,
                            self.attWUserUserInt[i], self.attBUserUserInt[i], self.attWUserUserInt1[i])
                        itemItemIntW1, esum = _attMLP(
                            tf.concat([itemItemIntNE, tf.tile(iE1, tf.stack([1, topN, 1]))], 2), itemItemIntNM,
                            self.attBItemItemInt[i], self.attBItemItemInt[i], self.attWItemItemInt1[i])

                    userUserIntNESum = tf.reduce_sum(userUserIntNE * userUserIntW1, axis=1)  # userNum*E
                    itemItemIntNESum = tf.reduce_sum(itemItemIntNE * itemItemIntW1, axis=1)  # ItemNum*E

                    zero_vector_u = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    userUserIntNESum = tf.concat([userUserIntNESum, zero_vector_u], 0)
                    zero_vector_i = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    itemItemIntNESum = tf.concat([itemItemIntNESum, zero_vector_i], 0)

                    # user-gram graph, item-item graph, neighbors' distance = 2
                    userUserSeq2NE = tf.gather(uEmbedTmp, self.userUserSeq2Matrix)  # userNum*N*E
                    itemItemSeq2NE = tf.gather(iEmbedTmp, self.itemItemSeq2Matrix)  # itemNum*N*E

                    userUserSeq2NM = tf.gather(self.userMask, self.userUserSeq2Matrix)  # userNum*N*1
                    itemItemSeq2NM = tf.gather(self.itemMask, self.itemItemSeq2Matrix)  # itemNum*N*1

                    if self.attMode == 0:
                        userUserSeq2W1, esum1 = _attMLP(uE1 * userUserSeq2NE, userUserSeq2NM, self.attWUserUserSeq2[i],
                                                        self.attBUserUserSeq2[i], self.attWUserUserSeq21[i])
                        itemItemSeq2W1, esum = _attMLP(iE1 * itemItemSeq2NE, itemItemSeq2NM, self.attWItemItemSeq2[i],
                                                       self.attBItemItemSeq2[i], self.attWItemItemSeq21[i])
                    else:
                        userUserSeq2W1, esum1 = _attMLP(
                            tf.concat([userUserSeq2NE, tf.tile(uE1, tf.stack([1, topN, 1]))], 2), userUserSeq2NM,
                            self.attWUserUserSeq2[i], self.attBUserUserSeq2[i], self.attBUserUserSeq21[i])
                        itemItemSeq2W1, esum = _attMLP(
                            tf.concat([itemItemSeq2NE, tf.tile(iE1, tf.stack([1, topN, 1]))], 2), itemItemSeq2NM,
                            self.attWItemItemSeq2[i], self.attBItemItemSeq2[i], self.attWItemItemSeq21[i])

                    userUserSeq2NESum = tf.reduce_sum(userUserSeq2NE * userUserSeq2W1, axis=1)  # userNum*E
                    itemItemSeq2NESum = tf.reduce_sum(itemItemSeq2NE * itemItemSeq2W1, axis=1)  # ItemNum*E

                    zero_vector_u = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    userUserSeq2NESum = tf.concat([userUserSeq2NESum, zero_vector_u], 0)
                    zero_vector_i = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    itemItemSeq2NESum = tf.concat([itemItemSeq2NESum, zero_vector_i], 0)

                    uEmbedTmp = tf.matmul(uEmbedTmp, self.userW[i]) \
                                + tf.matmul(userItemNESum, self.userItemNW[i]) + tf.matmul(userGramNESum, self.userGramNW[i]) \
                                + tf.matmul(userUserIntNESum, self.userUserIntNW[i]) + tf.matmul(userUserSeq2NESum, self.userUserSeq2NW[i])
                    iEmbedTmp = tf.matmul(iEmbedTmp, self.itemW[i]) \
                                + tf.matmul(itemUserNESum, self.itemUserNW[i]) + tf.matmul(itemItemNESum, self.itemItemNW[i]) \
                                + tf.matmul(itemItemIntNESum, self.itemItemIntNW[i]) + tf.matmul(itemItemSeq2NESum, self.itemItemSeq2NW[i])

                gEmbedTmp = self.gEmbed

        uE = tf.nn.embedding_lookup(uEmbedTmp, self.user)  # B*E
        iE = tf.nn.embedding_lookup(iEmbedTmp, self.item)  # B*E

        self.output = tf.reduce_sum(tf.multiply(uE, iE), 1)
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.regValue), tf.trainable_variables())
        self.cost = tf.losses.mean_squared_error(self.r, self.output) + reg
        self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def fit(self, currentData, epochNum):
        [user, item, rating] = currentData
        costAvg = 0
        for step in range(epochNum):
            cost, _ = self.sess.run([self.cost, self.optimize], feed_dict={self.user: user, self.item: item, self.r: rating})
            costAvg += cost
            # if step%4 == 0:
            #     self.saveModel('./model/pgcna/model.ckpt',step)
        return costAvg

    def saveModel(self, fileName, step):
        self.saver.save(self.sess, fileName, global_step=step)

    def restoreModel(self, fileName):
        self.saver.restore(self.sess, fileName)

    def predict(self, currentData):
        [user, item] = currentData
        pRating = self.sess.run(self.output,
                                feed_dict={self.user: user, self.item: item})
        return pRating

    def _getVar(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        variable = tf.Variable(initial, name=name)
        return variable

    def _getEmbed(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        variable = tf.Variable(initial, name=name)
        zero_vector = tf.constant(0.0, tf.float32, [1, shape[1]])
        var = tf.concat([variable,zero_vector],0)
        return var