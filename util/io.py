import os.path
from os import makedirs,remove
from re import compile,findall,split
from .config import LineConfig
class FileIO(object):
    def __init__(self):
        pass

    # @staticmethod
    # def writeFile(filePath,content,op = 'w'):
    #     reg = compile('(.+[/|\\\]).+')
    #     dirs = findall(reg,filePath)
    #     if not os.path.exists(filePath):
    #         os.makedirs(dirs[0])
    #     with open(filePath,op) as f:
    #         f.write(str(content))

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def loadDataSet(conf, file, bTest=False,binarized = False, threshold = 3.0):
        trainingData = []
        testData = []
        ratingConfig = LineConfig(conf['ratings.setup'])
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()#这个直接读取整个文件
        # ignore the headline
        if ratingConfig.contains('-header'):
            ratings = ratings[1:]
        # order of the columns
        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'
        if ratingConfig.contains('-delim'):
            delim=ratingConfig['-delim']
        for lineNo, line in enumerate(ratings):
            miRNAs = split(delim,line.strip())#[140,137,1]类似这种形式的列表
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                circRNAId = miRNAs[int(order[0])]
                miRNAId = miRNAs[int(order[1])]
                if len(order)<3:#如果不知道关联，默认为1
                    rating = 1 #default value
                else:
                    rating = miRNAs[int(order[2])]
                if binarized:
                    if float(miRNAs[int(order[2])])<threshold:
                        continue
                    else:
                        rating = 1
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if bTest:
                testData.append([circRNAId, miRNAId, float(rating)])
            else:
                trainingData.append([circRNAId, miRNAId, float(rating)])
        if bTest:
            return testData
        else:
            return trainingData

    @staticmethod
    def loadcircRNAList(filepath):
        circRNAList = []
        print('loading circRNA List...')
        with open(filepath) as f:
            for line in f:
                circRNAList.append(line.strip().split()[0])
        return circRNAList

    @staticmethod
    def loadRelationship(conf, filePath):
        socialConfig = LineConfig(conf['social.setup'])
        relation = []
        print('loading social data...')
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if socialConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = socialConfig['-columns'].strip().split()
        if len(order) <= 2:
            print('The social file is not in a correct format.')
        for lineNo, line in enumerate(relations):
            miRNAs = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The social file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            circRNAId1 = miRNAs[int(order[0])]
            circRNAId2 = miRNAs[int(order[1])]
            if len(order) < 3:
                weight = 1
            else:
                weight = float(miRNAs[int(order[2])])
            relation.append([circRNAId1, circRNAId2, weight])
        return relation




