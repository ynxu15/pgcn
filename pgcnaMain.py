import pickle
import numpy as np
from pgcna import PGCN_A
import os
from evaluate import *
from settings import *
from processData import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = Config('pgcna')

# notes: if you dont have large GPU memory, try small embedSize, less convolution layers, small distance

# These settings are for Retailrocket dataset
topN = 50
dataPath = './data/retail10_30_top%d/' %(topN)
resultFileName = 'retail10_30_pgcna.txt'
config.regValue = 8e-5
config.embedSize = 32  # 64 -> run out of memory


# These settings are for MovieLens dataset
# topN = 200
# dataPath = './data/ml100k_top%d/' % (topN)
# resultFileName = 'movielens100k_pgcna.txt'
# config.regValue = 8e-5
# config.convLayerNum = 1


config.topN = topN
detailMode = False              #Print detailed information
batchNum = 500
maxEpochNum = 100

batchIndex = 0
def getBatch(randomIndices, batchNum):
    '''Return a batch of indexs of training data'''
    global batchIndex
    low = batchNum*batchIndex
    if low+batchNum<randomIndices.shape[0]:
        batchIndex += 1
        return randomIndices[low: low+batchNum]
    else:
        batchIndex = 0
        return randomIndices[low:]

########################################################
'''Main function'''

print('reading training and testing data...' )
trainData, testData,testNegData, userItemDic, itemUserDic, userUserIntDic, itemItemIntDic, \
itemItemSeq1Dic, itemItemSeq2Dic, userGramSeq1Dic, userUserSeq2Dic, userNum, itemNum, gramNum = read_data(dataPath)
print('******************** Data Info *********************')
print(dataPath)
print('# of users: %d, # of items: %d' % (userNum, itemNum))
print('****************************************************')


print('building model')
pn = PGCN_A(userNum=userNum, itemNum=itemNum, gramNum=gramNum, config=config, initialize = True,
        userItemDic=userItemDic, itemUserDic=itemUserDic, userGramSeq1Dic=userGramSeq1Dic, itemItemSeq1Dic=itemItemSeq1Dic,
        userUserIntDic=userUserIntDic, itemItemIntDic=itemItemIntDic, userUserSeq2Dic=userUserSeq2Dic,itemItemSeq2Dic=itemItemSeq2Dic)
pn.printSettings()

(hits, ndcgs) = evaluate_model(pn, testData, testNegData, 10, 1)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('HR=%.6f, NDCG=%.6f'%(hr, ndcg))

trainUser, trainItem, trainRating = getTrainData(trainData, itemNum)
randomIndices = np.random.permutation(trainUser.shape[0])

resultFile = open(resultFileName,'w')
hitHis, ndcgHis = 0.0, 0.0          # historical results
costSum, costCount = 0.0, 0.00001
epoch = 0
while epoch < maxEpochNum:
    if batchIndex == 0:
        if epoch%1 == 0:
            (hits, ndcgs) = evaluate_model(pn, testData, testNegData, 10, 1)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('epoch:%d, loss=%.6f, HR=%.6f, NDCG=%.6f'%(epoch, costSum / costCount, hr, ndcg))
            if detailMode and (hitHis < hr or ndcgHis < ndcg):
                hitHis, ndcgHis = hr, ndcg
                for t in range(2, 11, 2):
                    (hits, ndcgs) = evaluate_model(pn, testData, testNegData, t, 1)
                    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                    resultFile.write('epoch:%d, loss=%.6f, HR=%.6f, NDCG=%.6f\n' % (epoch, costSum / costCount, hr, ndcg))
                resultFile.write('\n')
                resultFile.flush()
        costSum, costCount = 0.0, 0.00001
        epoch += 1

    dataIndex = getBatch(randomIndices, batchNum)
    currentData = [trainUser[dataIndex], trainItem[dataIndex], trainRating[dataIndex]]
    cost = pn.fit(currentData, 1)
    costSum += cost
    costCount += 1

    # if epoch % 4 == 0:
    #     pn.saveModel('./model/pgcna/model.ckpt', epoch)

num_thread, K = 1, 10
(hits, ndcgs) = evaluate_model(pn, testData, testNegData, K, num_thread)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('HR=%.6f, NDCG=%.6f'%(hr, ndcg))











