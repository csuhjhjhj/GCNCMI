from base.iterativeRecommender import IterativeRecommender
from random import shuffle,randint,choice
import tensorflow as tf



class DeepRecommender(IterativeRecommender):
    def __init__(self,conf,trainingSet,testSet,fold='[1]'):
        super(DeepRecommender, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(DeepRecommender, self).readConfiguration()
        # set the reduced dimension
        self.batch_size = int(self.config['batch_size'])

    def printAlgorConfig(self):
        super(DeepRecommender, self).printAlgorConfig()

    def initModel(self):
        super(DeepRecommender, self).initModel()
        self.u_idx = tf.placeholder(tf.int32, name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, name="v_idx")
        self.r = tf.placeholder(tf.float32, name="rating")
        self.circRNA_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_circRNAs, self.emb_size], stddev=0.005), name='U')
        self.miRNA_embeddings = tf.Variable(tf.truncated_normal(shape=[self.num_miRNAs, self.emb_size], stddev=0.005), name='V')
        self.u_embedding = tf.nn.embedding_lookup(self.circRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.miRNA_embeddings, self.v_idx)


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # print('config是什么:',config)
        self.sess = tf.Session(config=config)

    def next_batch_pairwise(self):
        shuffle(self.data.trainingData)
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                circRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.batch_size + batch_id)]
                miRNAs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                circRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                miRNAs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            miRNA_list = list(self.data.miRNA.keys())
            for i, circRNA in enumerate(circRNAs):
                i_idx.append(self.data.miRNA[miRNAs[i]])
                u_idx.append(self.data.circRNA[circRNA])
                neg_miRNA = choice(miRNA_list)
                while neg_miRNA in self.data.trainSet_u[circRNA]:
                    neg_miRNA = choice(miRNA_list)
                j_idx.append(self.data.miRNA[neg_miRNA])

            yield u_idx, i_idx, j_idx

    def next_batch_pointwise(self):
        batch_id=0
        while batch_id<self.train_size:
            if batch_id+self.batch_size<=self.train_size:
                circRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id,self.batch_size+batch_id)]
                miRNAs = [self.data.trainingData[idx][1] for idx in range(batch_id,self.batch_size+batch_id)]
                batch_id+=self.batch_size
            else:
                circRNAs = [self.data.trainingData[idx][0] for idx in range(batch_id, self.train_size)]
                miRNAs = [self.data.trainingData[idx][1] for idx in range(batch_id, self.train_size)]
                batch_id=self.train_size
            u_idx,i_idx,y = [],[],[]
            for i,circRNA in enumerate(circRNAs):
                i_idx.append(self.data.miRNA[miRNAs[i]])
                u_idx.append(self.data.circRNA[circRNA])
                y.append(1)
                for instance in range(4):
                    miRNA_j = randint(0, self.num_miRNAs - 1)
                    while self.data.id2miRNA[miRNA_j] in self.data.trainSet_u[circRNA]:
                        miRNA_j = randint(0, self.num_miRNAs - 1)
                    u_idx.append(self.data.circRNA[circRNA])
                    i_idx.append(miRNA_j)
                    y.append(0)
            yield u_idx,i_idx,y

    def predictForRanking(self,u):
        'used to rank all the miRNAs for the circRNA'
        pass


