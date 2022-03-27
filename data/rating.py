import numpy as np
from util.config import Config,LineConfig
import random
from collections import defaultdict
#rating类是吧我们的数据读进来封装成python的数据结构的一个类
class Rating(object):
    'data access control'
    def __init__(self,config,trainingSet, testSet):
        self.config = config
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.circRNA = {} #map circRNA names to id  774个circRNA
        self.miRNA = {} #map miRNA names to id   1929个miRNA
        self.id2circRNA = {}
        self.id2miRNA = {}
        self.circRNAMeans = {} #mean values of circRNAs's ratings
        self.miRNAMeans = {} #mean values of miRNAs's ratings
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict) #test set in the form of [circRNA][miRNA]=rating
        self.testSet_i = defaultdict(dict) #test set in the form of [miRNA][circRNA]=rating
        self.rScale = [] #rating scale
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computemiRNAMean()
        self.__computecircRNAMean()
        self.__globalAverage()


    def __generateSet(self):
        scale = set()
        #if validation is conducted, we sample the training data at a given probability to form the validation set,
        #and then replacing the test data with the validation data to tune parameters.
        if self.evalSettings.contains('-val'):
            random.shuffle(self.trainingData)
            separation = int(self.elemCount()*float(self.evalSettings['-val']))
            self.testData = self.trainingData[:separation]
            self.trainingData = self.trainingData[separation:]
        for i,entry in enumerate(self.trainingData):
            circRNAName,miRNAName,rating = entry
            # makes the rating within the range [0, 1].
            #rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            #self.trainingData[i][2] = rating
            # order the circRNA
            if circRNAName not in self.circRNA:
                self.circRNA[circRNAName] = len(self.circRNA)
                self.id2circRNA[self.circRNA[circRNAName]] = circRNAName
            # order the miRNA
            if miRNAName not in self.miRNA:
                self.miRNA[miRNAName] = len(self.miRNA)
                self.id2miRNA[self.miRNA[miRNAName]] = miRNAName
                # circRNAList.append
            self.trainSet_u[circRNAName][miRNAName] = rating
            self.trainSet_i[miRNAName][circRNAName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            if self.evalSettings.contains('-predict'):
                self.testSet_u[entry]={}
            else:
                circRNAName, miRNAName, rating = entry
                self.testSet_u[circRNAName][miRNAName] = rating
                self.testSet_i[miRNAName][circRNAName] = rating


    def __globalAverage(self):
        total = sum(self.circRNAMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.circRNAMeans)

    def __computecircRNAMean(self):
        for u in self.circRNA:
            self.circRNAMeans[u] = sum(self.trainSet_u[u].values())/len(self.trainSet_u[u])

    def __computemiRNAMean(self):
        for c in self.miRNA:
            self.miRNAMeans[c] = sum(self.trainSet_i[c].values())/len(self.trainSet_i[c])

    def getcircRNAId(self,u):
        if u in self.circRNA:
            return self.circRNA[u]

    def getmiRNAId(self,i):
        if i in self.miRNA:
            return self.miRNA[i]

    def trainingSize(self):
        return (len(self.circRNA),len(self.miRNA),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether circRNA u rated miRNA i'
        if u in self.circRNA and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containscircRNA(self,u):
        'whether circRNA is in training set'
        if u in self.circRNA:
            return True
        else:
            return False

    def containsmiRNA(self,i):
        'whether miRNA is in training set'
        if i in self.miRNA:
            return True
        else:
            return False

    def circRNARated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def miRNARated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())

    def row(self,u):
        k,v = self.circRNARated(u)
        vec = np.zeros(len(self.miRNA))
        #print vec
        for pair in zip(k,v):
            iid = self.miRNA[pair[0]]
            vec[iid]=pair[1]
        return vec

    def col(self,i):
        k,v = self.miRNARated(i)
        vec = np.zeros(len(self.circRNA))
        #print vec
        for pair in zip(k,v):
            uid = self.circRNA[pair[0]]
            vec[uid]=pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.circRNA),len(self.miRNA)))
        for u in self.circRNA:
            k, v = self.circRNARated(u)
            vec = np.zeros(len(self.miRNA))
            # print vec
            for pair in zip(k, v):
                iid = self.miRNA[pair[0]]
                vec[iid] = pair[1]
            m[self.circRNA[u]]=vec
        return m
    # def row(self,u):
    #     return self.trainingMatrix.row(self.getcircRNAId(u))
    #
    # def col(self,c):
    #     return self.trainingMatrix.col(self.getmiRNAId(c))

    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
