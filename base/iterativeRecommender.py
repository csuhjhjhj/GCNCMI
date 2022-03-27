from base.recommender import Recommender
from util import config
import numpy as np
from random import shuffle
from util.measure import Measure
from util.qmath import find_k_largest
class IterativeRecommender(Recommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(IterativeRecommender, self).__init__(conf,trainingSet,testSet,fold)
        self.bestPerformance = []

    def readConfiguration(self):
        super(IterativeRecommender, self).readConfiguration()
        # set the reduced dimension
        self.emb_size = int(self.config['num.factors'])
        # set maximum iteration
        self.maxIter = int(self.config['num.max.iter'])
        # set learning rate
        learningRate = config.LineConfig(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        if self.evalSettings.contains('-tf'):
            self.batch_size = int(self.config['batch_size'])
        # regularization parameter
        regular = config.LineConfig(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])

    def printAlgorConfig(self):
        super(IterativeRecommender, self).printAlgorConfig()
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Iteration:',self.maxIter)
        print('Regularization parameter: regU %.3f, regI %.3f, regB %.3f' %(self.regU,self.regI,self.regB))
        print('='*80)

    def initModel(self):
        self.P = np.random.rand(len(self.data.circRNA), self.emb_size) / 3 # latent circRNA matrix
        self.Q = np.random.rand(len(self.data.miRNA), self.emb_size) / 3  # latent miRNA matrix
        self.loss, self.lastLoss = 0, 0

    def buildModel_tf(self):
        # initialization
        import tensorflow as tf
        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.r = tf.placeholder(tf.float32, [None], name="rating")
        self.U = tf.Variable(tf.truncated_normal(shape=[self.num_circRNAs, self.emb_size], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.num_miRNAs, self.emb_size], stddev=0.005), name='V')
        self.circRNA_biases = tf.Variable(tf.truncated_normal(shape=[self.num_circRNAs, 1], stddev=0.005), name='U')
        self.miRNA_biases = tf.Variable(tf.truncated_normal(shape=[self.num_miRNAs, 1], stddev=0.005), name='U')
        self.circRNA_bias = tf.nn.embedding_lookup(self.circRNA_biases, self.u_idx)
        self.miRNA_bias = tf.nn.embedding_lookup(self.miRNA_biases, self.v_idx)
        self.circRNA_embedding = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.miRNA_embedding = tf.nn.embedding_lookup(self.V, self.v_idx)

    def updateLearningRate(self,iter):
        if iter > 1:
            if abs(self.lastLoss) > abs(self.loss):
                self.lRate *= 1.05
            else:
                self.lRate *= 0.5
        if self.lRate > self.maxLRate > 0:
            self.lRate = self.maxLRate

    def predictForRating(self, u, i):
        if self.data.containscircRNA(u) and self.data.containsmiRNA(i):
            return self.P[self.data.circRNA[u]].dot(self.Q[self.data.miRNA[i]])#做两个向量的点积得到circRNA和miRNA的相似度
        elif self.data.containscircRNA(u) and not self.data.containsmiRNA(i):
            return self.data.circRNAMeans[u]
        elif not self.data.containscircRNA(u) and self.data.containsmiRNA(i):
            return self.data.miRNAMeans[i]
        else:
            return self.data.globalMean

    def predictForRanking(self,u):
        'used to rank all the miRNAs for the circRNA'
        if self.data.containscircRNA(u):
            return self.Q.dot(self.P[self.data.circRNA[u]])
        else:
            return [self.data.globalMean]*self.num_miRNAs

    def isConverged(self,iter):
        from math import isnan
        if isnan(self.loss):
            print('Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!')
            exit(-1)
        deltaLoss = (self.lastLoss-self.loss)
        if self.ranking.isMainOn():
            print('%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f' \
                  %(self.algorName,self.foldInfo,iter,self.loss,deltaLoss,self.lRate))
            #measure = self.ranking_performance(iter)
        else:
            measure = self.rating_performance()
            print('%s %s iteration %d: loss = %.4f, delta_loss = %.5f learning_Rate = %.5f %5s %5s' \
                  % (self.algorName, self.foldInfo, iter, self.loss, deltaLoss, self.lRate, measure[0].strip()[:11], measure[1].strip()[:12]))
        #check if converged
        cond = abs(deltaLoss) < 1e-3
        converged = cond
        if not converged:
            self.updateLearningRate(iter)
        self.lastLoss = self.loss
        shuffle(self.data.trainingData)
        return converged

    def rating_performance(self):
        res = []
        for ind, entry in enumerate(self.data.testData):
            circRNA, miRNA, rating = entry
            # predict
            prediction = self.predictForRating(circRNA, miRNA)
            pred = self.checkRatingBoundary(prediction)
            res.append([circRNA,miRNA,rating,pred])
        self.measure = Measure.ratingMeasure(res)
        return self.measure

    def ranking_performance(self,iteration):
        #for quick evaluation, we only rank 2000 miRNAs
        #results of 1000 circRNAs would be evaluated
        N = 10
        recList = {}
        testSample = {}
        for circRNA in self.data.testSet_u:
            if len(testSample) == 1000:
                break
            if circRNA not in self.data.trainSet_u:
                continue
            testSample[circRNA] = self.data.testSet_u[circRNA]

        for circRNA in testSample:
            miRNASet = {}
            predictedmiRNAs = self.predictForRanking(circRNA)
            for id, rating in enumerate(predictedmiRNAs[:2000]):
                miRNASet[self.data.id2miRNA[id]] = rating
            ratedList, ratingList = self.data.circRNARated(circRNA)
            for miRNA in ratedList:
                if miRNA in miRNASet:
                    del miRNASet[miRNA]
            recList[circRNA] = find_k_largest(N,miRNASet)
        measure = Measure.rankingMeasure(testSample, recList, [10])
        if len(self.bestPerformance)>0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -=1
            if count<0:
                self.bestPerformance[1]=performance
                self.bestPerformance[0]=iteration
                self.saveModel()
        else:
            self.bestPerformance.append(iteration)
            performance = {}
            for m in measure[1:]:
                k,v = m.strip().split(':')
                performance[k]=float(v)
                self.bestPerformance.append(performance)
            self.saveModel()
        print('-'*120)
        print('Quick Ranking Performance '+self.foldInfo+' (Top-10 miRNA Recommendation On 1000 sampled circRNAs)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('iteration:',iteration,' | '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Precision'+':'+str(self.bestPerformance[1]['Precision'])+' | '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'MAP' + ':' + str(self.bestPerformance[1]['MAP']) + ' | '
        bp += 'MDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('iteration:',self.bestPerformance[0],bp)
        print('-'*120)
        return measure

