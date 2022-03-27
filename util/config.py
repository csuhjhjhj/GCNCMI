import os.path
#读取配置文件的一个类
class Config(object):
    def __init__(self,fileName):
        self.config = {}
        self.readConfiguration(fileName)

    def __getmiRNA__(self, miRNA):
        if not self.contains(miRNA):
            print('parameter '+miRNA+' is invalid!')
            exit(-1)
        return self.config[miRNA]

    def contains(self,key):
        return key in self.config
    #读取配置文件的函数
    def readConfiguration(self,file):
        if not os.path.exists(file):
            print('config file is not found!')
            raise IOError
        with open(file) as f:
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        key,value=line.strip().split('=')
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d'%(ind))

class LineConfig(object):
    def __init__(self,content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,miRNA in enumerate(self.line):
            if (miRNA.startswith('-') or miRNA.startswith('--')) and  not miRNA[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[miRNA] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[miRNA] = 1

    def __getmiRNA__(self, miRNA):
        if not self.contains(miRNA):
            print('parameter '+miRNA+' is invalid!')
            exit(-1)
        return self.options[miRNA]

    def keys(self):
        return self.options.keys()

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options


