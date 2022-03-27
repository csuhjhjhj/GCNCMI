from util.config import LineConfig
from util.dataSplit import *
from util.io import FileIO
from mod.GCNCMI import GCNCMI

#处理数据集的一个类
class dataProcess(object):
    def __init__(self,config, i):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.relation = []
        self.measure = []
        self.config =config#到这里config配置文件里面的东西就读进来了
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.data_path="dataset"
        #更换数据集从这里把路径改了就可以了
        train_path = f"{self.data_path}/train_{i}.txt"
        test_path =  f"{self.data_path}/test_{i}.txt"
    # train_path = "./dataset/lastfm/data5/ass.txt"
    # test_path = "./dataset/lastfm/data5/test.txt"
        binarized = False
        bottom = 0
        #这两行代码是从配置文件得到训练集和测试集
        self.trainingData = FileIO.loadDataSet(config, train_path, binarized=binarized, threshold=bottom)#data9 有7672个对作为训练集
        self.testData = FileIO.loadDataSet(config, test_path, bTest=True, binarized=binarized,
                                           threshold=bottom)#1917个队作为测试集
        print("data split")
        if self.config.contains('evaluation.setup'):
            self.evaluation = LineConfig(config['evaluation.setup'])
        if config.contains('social'):
            self.socialConfig = LineConfig(self.config['social.setup'])
            self.relation = FileIO.loadRelationship(config,self.config['social'])

    def execute(self, i):
        # try:
        #     importStr = 'from model.' + self.config['model.name'] + ' import ' + self.config['model.name']
        #     print("hejie", importStr)
        #     exec (importStr)   #importStr 的意思是执行它括号内的语句
        #     print('-----------------------')
        # except ImportError:
        #     # importStr = 'from model.' + self.config['model.name'] + ' import ' + self.config['model.name']
        #     # print("hejie",importStr)
        #     # exec(importStr)
        #     print("路径错误-------")

        if self.config.contains('social'):
            recommender = self.config['model.name']+'(self.config,self.trainingData,self.testData,self.relation)'
        else:
            recommender = self.config['model.name'] + '(self.config,self.trainingData,self.testData)'

        print('rcommender是:',recommender)
        eval(recommender).execute(i)
        #eval用于执行一个字符串表达式，并返回表达式的计算结果
