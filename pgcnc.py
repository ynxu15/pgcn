#Authors: Yanan Xu<ynxu15@gmail.com>
#License: GNU General Public License v2.0

import numpy as np
import tensorflow as tf

class PGCN_C(object):
    """Class for Path conditioned Graph Convolutional Network
    ------------
    Learning Shared Vertex Representation in Heterogeneous Graphs with
    Convolutional Networks for Recommendation. Yanan Xu, 2019.
    """
    def __init__(self, userNum, itemNum, gramNum, config, initialize=True,
                 userItemDic=None, itemUserDic=None, userGramSeq1Dic=None, itemItemSeq1Dic=None,
                 userUserIntDic=None, itemItemIntDic=None, userUserSeq2Dic=None, itemItemSeq2Dic=None
                 ):
        self.userNum, self.itemNum, self.gramNum = userNum, itemNum, gramNum
        self.embedSize, self.lr = config.embedSize, config.lr
        self.initialize = initialize

        self.topN = config.topN                     # the maximum number of neighbors for each vertex
        self.mode = config.mode                     # 0: traditional MF, 1: use only user-item graph, 2: use all graphs
        self.convLayerNum = config.convLayerNum     # Number of convolutional layers:  1, 2, 3
        self.dis = config.dis                       # the largest distance between neighbors: 1, 2
        self.regValue = config.regValue             # regularization weight

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
        print('pgcnc model:')
        print('topN: %d, embedSize: %d, mode: %d, convLayers: %d, dis: %d, regValue: %.6f'%(self.topN, self.embedSize, self.mode, self.convLayerNum, self.dis, self.regValue))
        print('*'*80)

    def get_graph_data(self):
        userNum, itemNum, gramNum = self.userNum, self.itemNum, self.gramNum
        embedSize, topN = self.embedSize, self.topN
        if self.dis>0 and self.mode>0:
            # user-item graph, neighbors' distance = 1,
            userItemMatrix = np.zeros((userNum, topN), dtype=int)
            itemUserMatrix = np.zeros((itemNum, topN), dtype=int)
            userItemMatrix[:, :], itemUserMatrix[:, :] = itemNum,userNum
            userItemWMatrix = np.zeros((userNum, topN), dtype=float)  # weight
            itemUserWMatrix = np.zeros((itemNum, topN), dtype=float)  # weight

            for u in self.userItemDic:
                ddic = self.userItemDic[u]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                userItemMatrix[u, :len(neighbor)] = neighbor
                userItemWMatrix[u, :len(neighbor)] = neighborW
            for i in self.itemUserDic:
                ddic = self.itemUserDic[i]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                itemUserMatrix[i, :len(neighbor)] = neighbor
                itemUserWMatrix[i, :len(neighbor)] = neighborW

            self.userItemMatrix = tf.constant(userItemMatrix, dtype=tf.int32,shape=(userNum, topN),name='userItemMatrix')
            self.itemUserMatrix = tf.constant(itemUserMatrix, dtype=tf.int32,shape=(itemNum, topN),name='itemUserMatrix')
            self.userItemWMatrix = tf.constant(userItemWMatrix, dtype= tf.float32,shape=(userNum, topN),name='userItemWMatrix')
            self.itemUserWMatrix = tf.constant(itemUserWMatrix, dtype=tf.float32,shape=(itemNum, topN),name='itemUserWMatrix')

            del userItemMatrix, itemUserMatrix, userItemWMatrix, itemUserWMatrix
            self.userItemDic, self.itemUserDic = [], []

        if self.dis>0 and self.mode >1:
            # user-gram graph, item-item graph, neighbors distance = 1,
            userGramMatrix = np.zeros((userNum, topN), dtype=int)
            itemItemMatrix = np.zeros((itemNum, topN), dtype=int)
            userGramMatrix[:, :], itemItemMatrix[:, :] = gramNum, itemNum
            userGramWMatrix = np.zeros((userNum, topN), dtype=float)  # weight
            itemItemWMatrix = np.zeros((itemNum, topN), dtype=float)  # weight

            for u in self.userGramSeq1Dic:
                ddic = self.userGramSeq1Dic[u]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                userGramMatrix[u, :len(neighbor)] = neighbor
                userGramWMatrix[u, :len(neighbor)] = neighborW
            for i in self.itemItemSeq1Dic:
                ddic = self.itemItemSeq1Dic[i]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                itemItemMatrix[i, :len(neighbor)] = neighbor
                itemItemWMatrix[i, :len(neighbor)] = neighborW

            self.userGramMatrix = tf.constant(userGramMatrix, dtype=tf.int32,shape=(userNum, topN),name='userItemMatrix')
            self.itemItemMatrix = tf.constant(itemItemMatrix, dtype=tf.int32,shape=(itemNum, topN),name='itemUserMatrix')
            self.userGramWMatrix = tf.constant(userGramWMatrix, dtype= tf.float32,shape=(userNum, topN),name='userItemWMatrix')
            self.itemItemWMatrix = tf.constant(itemItemWMatrix, dtype=tf.float32,shape=(itemNum, topN),name='itemUserWMatrix')

            del userGramMatrix, itemItemMatrix, userGramWMatrix, itemItemWMatrix
            self.userGramSeq1Dic, self.itemItemSeq1Dic  = [], []

        if self.dis > 1 and self.mode > 0:
            # user-item graph, neighbors' distance = 2,
            userUserIntMatrix = np.zeros((userNum, topN), dtype=int)
            itemItemIntMatrix = np.zeros((itemNum, topN), dtype=int)
            userUserIntMatrix[:, :], itemItemIntMatrix[:, :] = userNum,itemNum
            userUserIntWMatrix = np.zeros((userNum, topN), dtype=float)  # weight
            itemItemIntWMatrix = np.zeros((itemNum, topN), dtype=float)  # weight

            for u in self.userUserIntDic:
                ddic = self.userUserIntDic[u]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                userUserIntMatrix[u, :len(neighbor)] = neighbor
                userUserIntWMatrix[u, :len(neighbor)] = neighborW
            for i in self.itemItemIntDic:
                mydic = self.itemItemIntDic[i]
                neighbor = list(mydic.keys())
                neighborW = list(mydic.values())
                itemItemIntMatrix[i, :len(neighbor)] = neighbor
                itemItemIntWMatrix[i, :len(neighbor)] = neighborW

            self.userUserIntMatrix = tf.constant(userUserIntMatrix, dtype=tf.int32,shape=(userNum, topN),name='userUserIntMatrix')
            self.itemItemIntMatrix = tf.constant(itemItemIntMatrix, dtype=tf.int32,shape=(itemNum, topN),name='itemItemIntMatrix')
            self.userUserIntWMatrix = tf.constant(userUserIntWMatrix, dtype= tf.float32,shape=(userNum, topN),name='userUserIntWMatrix')
            self.itemItemIntWMatrix = tf.constant(itemItemIntWMatrix, dtype=tf.float32,shape=(itemNum, topN),name='itemItemIntWMatrix')

            del userUserIntMatrix, itemItemIntMatrix, userUserIntWMatrix, itemItemIntWMatrix
            self.userUserIntDic, self.itemItemIntDic = [], []

        if self.dis>1 and self.mode>1:
            # user-gram graph, item-item graph, neighbors' distance = 2,
            userUserSeq2Matrix = np.zeros((userNum, topN), dtype=int)
            itemItemSeq2Matrix = np.zeros((itemNum, topN), dtype=int)
            userUserSeq2Matrix[:, :], itemItemSeq2Matrix[:, :] = userNum, itemNum
            userUserSeq2WMatrix = np.zeros((userNum, topN), dtype=float)  # weight
            itemItemSeq2WMatrix = np.zeros((itemNum, topN), dtype=float)  # weight

            for u in self.userUserSeq2Dic:
                ddic = self.userUserSeq2Dic[u]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                userUserSeq2Matrix[u, :len(neighbor)] = neighbor
                userUserSeq2WMatrix[u, :len(neighbor)] = neighborW
            for i in self.itemItemSeq2Dic:
                ddic = self.itemItemSeq2Dic[i]
                neighbor = list(ddic.keys())
                neighborW = list(ddic.values())
                itemItemSeq2Matrix[i, :len(neighbor)] = neighbor
                itemItemSeq2WMatrix[i, :len(neighbor)] = neighborW

            self.userUserSeq2Matrix = tf.constant(userUserSeq2Matrix, dtype=tf.int32,shape=(userNum, topN),name='userUserSeq2Matrix')
            self.itemItemSeq2Matrix = tf.constant(itemItemSeq2Matrix, dtype=tf.int32,shape=(itemNum, topN),name='itemItemSeq2Matrix')
            self.userUserSeq2WMatrix = tf.constant(userUserSeq2WMatrix, dtype= tf.float32,shape=(userNum, topN),name='userUserSeq2WMatrix')
            self.itemItemSeq2WMatrix = tf.constant(itemItemSeq2WMatrix, dtype=tf.float32,shape=(itemNum, topN),name='itemItemSeq2WMatrix')

            del userUserSeq2Matrix, itemItemSeq2Matrix, userUserSeq2WMatrix, itemItemSeq2WMatrix
            self.userUserSeq2Dic, self.itemItemSeq2Dic  = [], []

    def initialize_graph(self):
        self._setup_base_graph()                              # setup the graph
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
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

        if self.mode > 0:
            self.userW, self.itemW, self.userItemNW, self.itemUserNW  = [], [], [], []
            for i in range(self.convLayerNum):
                self.userW.append(_getVar((embedSize, embedSize), 'userW%d'%(i)))
                self.itemW.append(_getVar((embedSize, embedSize), 'itemW%d'%(i)))
                self.userItemNW.append(_getVar((embedSize, embedSize), 'userItemNW%d'%(i)))
                self.itemUserNW.append(_getVar((embedSize, embedSize), 'itemUserNW%d'%(i)))

            if self.dis>1:
                self.userUserIntNW, self.itemItemIntNW = [], []
                for i in range(self.convLayerNum):
                    self.userUserIntNW.append(
                        _getVar((embedSize, embedSize), 'userUserIntNW%d' % (i)))
                    self.itemItemIntNW.append(
                        _getVar((embedSize, embedSize), 'itemItemIntNW%d' % (i)))
        if self.mode >1:
            self.gEmbed = self._getEmbed((gramNum, embedSize), 'gramEmbedding')
            self.userGramNW, self.itemItemNW = [], []
            for i in range(self.convLayerNum):
                self.userGramNW.append(_getVar((embedSize, embedSize), 'userGramNW%d'%(i)))
                self.itemItemNW.append(_getVar((embedSize, embedSize), 'itemItemNW%d'%(i)))

            if self.dis>1:
                self.userUserSeq2NW, self.itemItemSeq2NW = [], []
                for i in range(self.convLayerNum):
                    self.userUserSeq2NW.append(
                        _getVar((embedSize, embedSize), 'userUserSeq2NW%d' % (i)))
                    self.itemItemSeq2NW.append(
                        _getVar((embedSize, embedSize), 'itemItemSeq2NW%d' % (i)))

        self.saver = tf.train.Saver(max_to_keep=10)

    def _setup_training(self):
        embedSize, topN = self.embedSize, self.topN
        uEmbedTmp, iEmbedTmp = self.uEmbed, self.iEmbed

        if self.mode == 0:
            pass

        if self.mode == 1:
            for i in range(self.convLayerNum):

                # neighbors' distance = 1
                userItemNE = tf.gather(iEmbedTmp, self.userItemMatrix)  # userNum*N*E
                itemUserNE = tf.gather(uEmbedTmp, self.itemUserMatrix)  # itemNum*N*E

                userItemNESum = tf.reduce_sum(userItemNE * tf.reshape(self.userItemWMatrix, (-1, topN, 1)), axis=1)  # userNum*E
                itemUserNESum = tf.reduce_sum(itemUserNE * tf.reshape(self.itemUserWMatrix, (-1, topN, 1)), axis=1)  # ItemNum*E

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

                    userUserIntNESum = tf.reduce_sum(userUserIntNE * tf.reshape(self.userUserIntWMatrix, (-1, topN, 1)),
                                                     axis=1)  # userNum*E
                    itemItemIntNESum = tf.reduce_sum(itemItemIntNE * tf.reshape(self.itemItemIntWMatrix, (-1, topN, 1)),
                                                     axis=1)  # ItemNum*E

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

                userItemNESum = tf.reduce_sum(userItemNE * tf.reshape(self.userItemWMatrix, (-1, topN, 1)),
                                              axis=1)  # userNum*E
                itemUserNESum = tf.reduce_sum(itemUserNE * tf.reshape(self.itemUserWMatrix, (-1, topN, 1)),
                                              axis=1)  # ItemNum*E

                userGramNESum = tf.reduce_sum(userGramNE * tf.reshape(self.userGramWMatrix, (-1, topN, 1)), axis=1)  # userNum*E
                itemItemNESum = tf.reduce_sum(itemItemNE * tf.reshape(self.itemItemWMatrix, (-1, topN, 1)), axis=1)  # itemNum*E

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
                    # neighbors with distance = 2, user-item graph
                    userUserIntNE = tf.gather(uEmbedTmp, self.userUserIntMatrix)  # userNum*N*E
                    itemItemIntNE = tf.gather(iEmbedTmp, self.itemItemIntMatrix)  # itemNum*N*E

                    userUserIntNESum = tf.reduce_sum(userUserIntNE * tf.reshape(self.userUserIntWMatrix, (-1, topN, 1)),
                                                     axis=1)  # userNum*E
                    itemItemIntNESum = tf.reduce_sum(itemItemIntNE * tf.reshape(self.itemItemIntWMatrix, (-1, topN, 1)),
                                                     axis=1)  # ItemNum*E

                    zero_vector_u = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    userUserIntNESum = tf.concat([userUserIntNESum, zero_vector_u], 0)
                    zero_vector_i = tf.constant([0.0] * embedSize, tf.float32, [1, embedSize])
                    itemItemIntNESum = tf.concat([itemItemIntNESum, zero_vector_i], 0)

                    # user-gram graph, item-item graph, neighbors' distance = 2,
                    userUserSeq2NE = tf.gather(uEmbedTmp, self.userUserSeq2Matrix)  # userNum*N*E
                    itemItemSeq2NE = tf.gather(iEmbedTmp, self.itemItemSeq2Matrix)  # itemNum*N*E

                    userUserSeq2NESum = tf.reduce_sum(userUserSeq2NE * tf.reshape(self.userUserSeq2WMatrix, (-1, topN, 1)),
                                                     axis=1)  # userNum*E
                    itemItemSeq2NESum = tf.reduce_sum(itemItemSeq2NE * tf.reshape(self.itemItemSeq2WMatrix, (-1, topN, 1)),
                                                     axis=1)  # ItemNum*E

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


    def fit(self, currentData, iterNum):
        [user, item, rating] = currentData
        costAvg = 0
        for step in range(iterNum):
            cost, _ = self.sess.run([self.cost, self.optimize], feed_dict={self.user:user, self.item:item,self.r: rating})
            costAvg += cost
            # if step%4 == 0:
            #     self.saveModel('./model/pgcnc/model.ckpt',step)
        return costAvg

    def saveModel(self, fileName, step):
        self.saver.save(self.sess, fileName, global_step=step)

    def restoreModel(self, fileName):
        self.saver.restore(self.sess, fileName)

    def predict(self, currentData):
        [user, item] = currentData
        pRating = self.sess.run(self.output, feed_dict={self.user: user, self.item: item})
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
