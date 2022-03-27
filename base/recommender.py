from data.rating import Rating
from util.io import FileIO
from util.config import LineConfig
from util.log import Log
from os.path import abspath
from time import strftime,localtime,time
from util.measure import Measure
from util.qmath import find_k_largest
from sklearn.metrics import roc_auc_score, precision_recall_curve,auc
import numpy as np
import random
from operator import itemgetter

sumauc = []
sumaucpr = []

class Recommender(object):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        self.config = conf
        self.data = None
        self.isSaveModel = False
        self.ranking = None
        self.isLoadModel = False
        self.output = None
        self.isOutput = True
        self.rankingNum=50
        self.data = Rating(self.config, trainingSet, testSet)
        self.foldInfo = fold
        self.evalSettings = LineConfig(self.config['evaluation.setup'])
        self.measure = []
        self.recOutput = []
        self.data_path="dataset"
        if self.evalSettings.contains('-cold'):
            #evaluation on cold-start circRNAs
            threshold = int(self.evalSettings['-cold'])
            removedcircRNA = {}
            for circRNA in self.data.testSet_u:
                if circRNA in self.data.trainSet_u and len(self.data.trainSet_u[circRNA])>threshold:
                    removedcircRNA[circRNA]=1
            for circRNA in removedcircRNA:
                del self.data.testSet_u[circRNA]
            testData = []
            for miRNA in self.data.testData:
                if miRNA[0] not in removedcircRNA:
                    testData.append(miRNA)
            self.data.testData = testData

        self.num_circRNAs, self.num_miRNAs, self.train_size = self.data.trainingSize()

    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.algorName,self.algorName+self.foldInfo+' '+currentTime)
        #save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k+'='+self.config[k])

    def readConfiguration(self):#读取配置
        self.algorName = self.config['model.name']
        self.output = LineConfig(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.ranking = LineConfig(self.config['miRNA.ranking'])

    def printAlgorConfig(self):#输出算法的配置
        "show model's configuration"
        print('Algorithm:',self.config['model.name'])
        print('Ratings dataset:',abspath(self.config['ratings']))
        if LineConfig(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:',abspath(LineConfig(self.config['evaluation.setup'])['-testSet']))
        #print dataset statistics
        print('Training set size: (circRNA count: %d, miRNA count %d, record count: %d)' %(self.data.trainingSize()))
        print('Test set size: (circRNA count: %d, miRNA count %d, record count: %d)' %(self.data.testSize()))
        print('='*80)
        #print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = LineConfig(self.config[self.config['model.name']])
            for key in args.keys():
                parStr+=key[1:]+':'+args[key]+'  '
            print('Specific parameters:',parStr)
            print('=' * 80)

    def initModel(self):#
        pass

    def buildModel(self):
        'build the model (for model-based algorithms )'
        pass

    def buildModel_tf(self):
        'training model on tensorflow'
        pass

    def saveModel(self):#保存模型
        pass

    def loadModel(self):#加载loadModel
        pass

    #for rating prediction
    def predictForRating(self, u, i):
        pass

    #for miRNA prediction
    def predictForRanking(self,u):
        pass

    def checkRatingBoundary(self,prediction):
        if prediction > self.data.rScale[-1]:
            return self.data.rScale[-1]
        elif prediction < self.data.rScale[0]:
            return self.data.rScale[0]
        else:
            return round(prediction,3)

    def evalRatings(self):
        res = list() #used to contain the text of the result
        res.append('circRNAId  miRNAId  original  prediction\n')
        #predict
        for ind,entry in enumerate(self.data.testData):
            circRNA,miRNA,rating = entry
            #predict
            prediction = self.predictForRating(circRNA, miRNA)
            #denormalize
            #prediction = denormalize(prediction,self.data.rScale[-1],self.data.rScale[0])
            #####################################
            pred = self.checkRatingBoundary(prediction)
            # add prediction in order to measure
            self.data.testData[ind].append(pred)
            res.append(circRNA+' '+miRNA+' '+str(rating)+' '+str(pred)+'\n')
        currentTime = strftime("%Y-%m-%d %H-%M-%S",localtime(time()))
        #output prediction result
        if self.isOutput:
            outDir = self.output['-dir']
            fileName = self.config['model.name']+'@'+currentTime+'-rating-predictions'+self.foldInfo+'.txt'
            FileIO.writeFile(outDir,fileName,res)
            print('The result has been output to ',abspath(outDir),'.')
        #output evaluation result
        outDir = self.output['-dir']
        fileName = self.config['model.name'] + '@'+currentTime +'-measure'+ self.foldInfo + '.txt'
        self.measure = Measure.ratingMeasure(self.data.testData)
        FileIO.writeFile(outDir, fileName, self.measure)
        self.log.add('###Evaluation Results###')
        self.log.add(self.measure)
        print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))
    #
    def check(self):
        pass


    def evalRanking(self, j):

        recList = {}
        circRNACount = len(self.data.testSet_u)#494
        print("circRNACount", self.data.testSet_u)
        #rawRes = {}
        # print("uuuuuuu", len(self.data.trainSet_i.keys()))
        predict_list = []
        actual_list = []
        path1 = f"{self.data_path}/ass.txt"
        path2 = f"{self.data_path}/test_{j}.txt"
        ass = {}#从ass.txt中得到一个与每个circRNA相关联的所有的miRNA的一个列表
        ass1 = {}#从test.txt中得到与每个circRNA相关联的所有的miRNA的一个列表
        ass2 ={}#从ass.txt中得到的与每个miRNA有关联的所有的circRNA的列表
        with open(path1) as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                if line[0] in ass:
                    ass[line[0]].append(int(line[1]))
                else:
                    ass[line[0]] = [int(line[1])]

                if line[1] in ass2:
                    ass2[int(line[1])].append(int(line[0]))
                else:
                    ass2[int(line[1])]=[int(line[0])]
        # with open(path2) as f:
        #     for line in f.readlines():
        #         line = line.strip().split('\t')
        #         if line[0] in ass:
        #             ass[line[0]].append(int(line[1]))
        #         else:
        #             ass[line[0]] = [int(line[1])]
        #
        with open(path2) as f:#打开用来进行交叉验证的数据
            for line in f.readlines():
                line = line.strip().split('\t')
                # print(line)
                if line[0] in ass1:
                    ass1[line[0]].append(int(line[1]))
                else:
                    ass1[line[0]] = [int(line[1])]
        negative_dict = {}
        for key, value in ass1.items():
            lg = len(value)
            all_i = set(ass[key])
            cu_i = set([i for i in range(0, 2115)])
            need_i = list(cu_i - all_i)
            random.shuffle(need_i)
            need_i = need_i[:lg]
            negative_dict[key] = need_i
        # print(negative_dict)
        # #

        # path2 = "./dataset/lastfm/data2/neg_4.txt"
        # negative_dict = {}
        # with open(path2) as f:
        #     for line in f.readlines():
        #         line = line.strip().split('\t')
        #         if line[0] in negative_dict:
        #             negative_dict[line[0]].append(line[1])
        #         else:
        #             negative_dict[line[0]] = [line[1]]

        # path1 = "./dataset/lastfm/data/test_4.txt"
        #
        # neg_sample = {}
        # with open(path1) as f:
        #     for line in f.readlines():
        #         line = line.strip().split("\t")
        #         circRNA = line[0]
        #         negative_miRNA = line[2]
        #         if circRNA in neg_sample:
        #             neg_sample[circRNA].append(negative_miRNA)
        #         else:
        #             neg_sample[circRNA] = [negative_miRNA]
        # print(neg_sample)

        result_dict = {}


        rankres1=[]

        # rankres2 = []
        # for i,miRNA in enumerate(self.data.testSet_i):
        #     predictedcircRNAs = self.predictForRanking(miRNA)
        #
        #     rank2=[]
        #     has_ass2=ass2[miRNA]
        #     for id,rating in enumerate(predictedcircRNAs):
        #         x=self.data.id2miRNA[id]
        #         if has_ass2.count(x)==0:
        #             rank2.append((int(x),float(rating)))
        #
        #     rankres2.append(rank2)

        predictcircRNAs={}#得到与每个miRNA的用户的评分
        for i, circRNA in enumerate(self.data.testSet_u):
            miRNASet = {}
            predictedmiRNAs = self.predictForRanking(circRNA)#得到的是一个与当前的circRNA相关的所有的miRNA的相关度


            # print("ass[circRNA]是",ass[circRNA])


            rank=[]

            has_ass=ass[circRNA]#得到与这个circRna已经有关联的miRNA
            rank.append(int(circRNA))
            for id,rating in enumerate(predictedmiRNAs):
                x=self.data.id2miRNA[id]
                if has_ass.count(x)==0:
                    rank.append((int(x),float(rating)))
                if int(x) in predictcircRNAs:
                    predictcircRNAs[int(x)].append((int(circRNA),float(rating)))
                else:
                    predictcircRNAs[int(x)]=[(int(circRNA),float(rating))]
            #这里predictcircRNAs由于是类似一个反向哈希表的构建过程,所以这里会发现每一个miRNA  对应的用户的编号顺序是对应的,但是关系程度是不一样的


            rankres1.append(rank)









            for id, rating in enumerate(predictedmiRNAs):

                miRNASet[self.data.id2miRNA[id]] = rating
            #miRNASet求出与当前用户有关的所有的商品的相关程度

        #     x = []
        #     for k in range(0, len(miRNASet.keys())):
        #         x.append(miRNASet[str(k)])
        #     result_dict[circRNA] = x
        # miRNA_dict = {}
        # path = "./dataset/lastfm/data5/ass.txt"
        # # path = "./dataset/lastfm/relation_index_split.txt"
        # ass_dict = {}
        # with open(path) as f:
        #     for line in f.readlines():
        #         line = line.strip().split('\t')
        #         # print(line)
        #         if line[1] in ass_dict:
        #             ass_dict[line[1]].append(line[0])
        #         else:
        #             ass_dict[line[1]] = [line[0]]
        # #
        # for i in range(0,1815):
        #     x = {}
        #     for j in range(0, 978):
        #         if str(j) in ass_dict[str(i)]:
        #             continue
        #         x[str(j)] = result_dict[str(j)][i]
        #     x = sorted(x.miRNAs(), key= lambda k: k[1], reverse=True)
        #     # print(x)
        #
        #     miRNA_dict[str(i)] = x[:100]
        # # #
        # np.save("./dataset/lastfm/data5/result_100.npy", miRNA_dict)
        #
            for i in self.data.testSet_u[circRNA].keys():
                if i in miRNASet:
                    va = miRNASet[i]#miRNASet求出用户与当前商品的预测关系值
                    actual_list.append(1)
                    predict_list.append(va)
                else:
                    actual_list.append(1)
                    predict_list.append(sum(miRNASet.values())/len(miRNASet.keys()))
                    # predict_list.append(1)
            for i in negative_dict[circRNA]:
                i = str(i)
                if i in miRNASet:

                    va = miRNASet[i]
                    actual_list.append(0)
                    predict_list.append(va)
                else:
                    actual_list.append(0)
                    predict_list.append(sum(miRNASet.values()) / len(miRNASet.keys()))
                    # predict_list.append(0)


         #调用将每一个circRNA的前50个
        self.saveRank(rankres1)
        self.saveRankmiRNA(predictcircRNAs,ass2)




        print("actual_list如下:")
        print(len(actual_list))
        print(actual_list)
        auc1 = roc_auc_score(actual_list, predict_list)

        print("predict_list如下:")
        print(len(predict_list))
        print(predict_list)


        sumauc.append(auc1)

        # if j==0:
        #     sumauc = auc1
        # else:
        #     sumauc+=auc1



        print('auc', auc1)

        precision, recall, _ = precision_recall_curve(actual_list, predict_list)
        AUCPR = auc(recall, precision)

        sumaucpr.append(AUCPR)




        # if j==0:
        #     sumaucpr = AUCPR
        # else:
        #     sumaucpr +=AUCPR


        print("AUPR", AUCPR)



        if j==4:
            print('五折交叉验证AUPR均值',(np.mean(sumaucpr)),'五折交叉验证auc均值',np.mean(sumauc))

        # print("调用recommender.py")

        # np.save(f'em512_layer2_k{j}.npy',[actual_list,predict_list])
        # print('factors_256',actual_list,predict_list)


        # path = "./dataset/lastfm/data4/measure_4.npy"
        # np.save(path, [actual_list, predict_list])

            # print("iddd", len(self.data.id2miRNA.keys()))
            # ratedList, ratingList = self.data.circRNARated(circRNA)
            # for miRNA in ratedList:
            #     del miRNASet[miRNA]
            # recList[circRNA] = find_k_largest(N,miRNASet)
            # if i % 100 == 0:
            #     print(self.algorName, self.foldInfo, 'progress:' + str(i) + '/' + str(circRNACount))
            # for miRNA in recList[circRNA]:
            #     line += ' (' + miRNA[0] + ',' + str(miRNA[1]) + ')'
            #     if miRNA[0] in self.data.testSet_u[circRNA]:
            #         line += '*'
            # line += '\n'
            #
            # self.recOutput.append(line)
        # currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        # if self.isOutput:
        #     outDir = self.output['-dir']
        #     fileName = self.config['model.name'] + '@' + currentTime + '-top-' + str(
        #     N) + 'miRNAs' + self.foldInfo + '.txt'
        #     FileIO.writeFile(outDir, fileName, self.recOutput)
        #     print('The result has been output to ', abspath(outDir), '.')
        # output evaluation result
        # if self.evalSettings.contains('-predict'):
        #     #no evalutation
        #     exit(0)
        # outDir = self.output['-dir']
        # fileName = self.config['model.name'] + '@' + currentTime + '-measure' + self.foldInfo + '.txt'
        # self.measure = Measure.rankingMeasure(self.data.testSet_u, recList, top)
        # self.log.add('###Evaluation Results###')
        # self.log.add(self.measure)
        # FileIO.writeFile(outDir, fileName, self.measure)
        # print('The result of %s %s:\n%s' % (self.algorName, self.foldInfo, ''.join(self.measure)))
    def saveRankmiRNA(self,arr,ass2):

        res=[]
        for key,value in arr.items():
            ans=[]
            ans.append(key)
            has_ass = ass2[key]


            for itm in value:
                if itm[0] in has_ass:
                    # print("itm[0] in has_ass",itm[0])
                    value.remove((itm[0],itm[1]))#在ass中为1，出现过，删除这个元素

            # print("key是:  ",key,"value是:",value)
            for x in value:
                ans.append(x)


            arr[key]=value
            res.append(ans)
        # arr=list(arr.values())

        # print(arr)

        self.saveRank(res,1)#1是一个标记值,为了搞清楚是在求ciRNA的前50个miRNA还是miRNA的前50个circRNA




    def saveRank(self,arr,flag=0):


        arr=sorted(arr,key=itemgetter(0))


        res=[]

        for line in arr:
            # print(line[0])
            rankans = []
            # if flag==1:
            #     rankans.append(line[0])
            # else:
            rankans.append(line[0])

            tempArr=line[1:]

            # sorted(tempArr,key=(lambda x:x[0]),reverse=True)
            tempArr=sorted(tempArr, key=itemgetter(1),reverse=True)

            line[1:]=tempArr



            cnt=self.rankingNum
            mincnt=len(tempArr)
            mincnt=min(mincnt,cnt)

            for i in range(1,mincnt+1):
                rankans.append(tempArr[i][0])

            res.append(rankans)

        #备注  这里self.data_path是我把原来的数据集中第一列和第二列换了位置，这样求出的就是miRNA有关的前50个circRNA

        # np.set_printoptions(suppress=True)
        # np.set_printoptions(precision=1)  # 设精度

        x=self.list_to_dict(arr)
        print(type(x))

        # if flag==1:
        #     np.save("CaseAnalysis/circRNA--50.npy",x)
        # else:
        #     np.save('CaseAnalysis/miRNA--50.npy',x)




        # if flag==0:
        #     np.savetxt("CaseAnalysis/circRNA-50.csv", res, fmt='%.00f', delimiter=',')  # 保留成整数
        # else:
        #     np.savetxt("CaseAnalysis/miRNA-50.csv", res, fmt='%s', delimiter=',')  # 保留成整数
        # print(res)


    def list_to_dict(self,arr):
        temp={}
        for line in arr:
            ans=[]
            for j in range(1,len(line)):
                ans.append(line[j])
            temp[line[0]]=ans
        return temp

    def execute(self, i):
        self.readConfiguration()
        print("dayungg")
        self.initializing_log()
        if self.foldInfo == '[1]':
            self.printAlgorConfig()
        #load model from disk or build model
        if self.isLoadModel:
            print('Loading model %s...' %self.foldInfo)
            self.loadModel()
        else:
            print('Initializing model %s...' %self.foldInfo)
            self.initModel()
            print('Building Model %s...' %self.foldInfo)
            try:
                if self.evalSettings.contains('-tf'):
                    import tensorflow
                    self.buildModel_tf()
                else:
                    self.buildModel()
            except ImportError:
                self.buildModel()
        #rating prediction or miRNA ranking
        print('Predicting %s...' %self.foldInfo)
        if self.ranking.isMainOn():
            self.evalRanking(i)
        else:
            self.evalRatings()
        #save model
        if self.isSaveModel:
            print('Saving model %s...' %self.foldInfo)
            self.saveModel()
        return self.measure





