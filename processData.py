import pickle
import numpy as np

def read_data(dataPath):
    inFile = open(dataPath + 'data.pkl','rb')
    data = pickle.load(inFile)

    # number of users, items, and subseqs, and user preference data
    userNum, itemNum, gramNum = data['userNum'], data['itemNum'], data['gramNum']
    trainData, testData, testNegData,  = data['trainData'], data['testData'], data['testNegData']

    # user-item graph data: user->item, distance = 1; item->user, distance = 1;
    userItemDic, itemUserDic  = data['userItemDic'], data['itemUserDic']
    # user->user, distance = 2; item->item, distance = 2;
    userUserIntDic, itemItemIntDic = data['userUserIntDic'], data['itemItemIntDic']

    # item-item graph data: item->item, distance = 1; item->item, distance = 2
    itemItemSeq1Dic, itemItemSeq2Dic  = data['itemItemSeq1Dic'], data['itemItemSeq2Dic']

    # user-subseq graph data: user->subseq, distance = 1; user->user, distance = 2
    userGramSeq1Dic, userUserSeq2Dic = data['userGramSeq1Dic'], data['userUserSeq2Dic']

    inFile.close()
    return trainData, testData, testNegData, userItemDic, itemUserDic, userUserIntDic, itemItemIntDic, itemItemSeq1Dic, itemItemSeq2Dic, userGramSeq1Dic, userUserSeq2Dic, userNum, itemNum, gramNum


def getTrainData(trainData, itemNum):
    trainUser, trainItem, trainRating = [], [], []
    negNum = 4
    for u,i in trainData:
        rating = 1.0
        trainUser.append(u)
        trainItem.append(i)
        trainRating.append(rating)

        # sample negative instances, set rating to 0
        for j in range(negNum):
            item2 = np.random.randint(itemNum)
            while (u,item2) in trainData or item2 == i:
                item2 = np.random.randint(itemNum)
            rating = 0.0
            trainUser.append(u)
            trainItem.append(item2)
            trainRating.append(rating)

    return np.array(trainUser), np.array(trainItem), np.array(trainRating)