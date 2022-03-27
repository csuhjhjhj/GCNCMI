#coding:utf8
from base.deepRecommender import DeepRecommender
from random import choice
import tensorflow as tf
import numpy as np
from math import sqrt


class GCNCMI(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(GCNCMI, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(GCNCMI, self).initModel()
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        ego_embeddings = tf.concat([self.circRNA_embeddings,self.miRNA_embeddings], axis=0)
        indices = [[self.data.circRNA[miRNA[0]],self.num_circRNAs+self.data.miRNA[miRNA[1]]] for miRNA in self.data.trainingData]
        indices += [[self.num_circRNAs+self.data.miRNA[miRNA[1]],self.data.circRNA[miRNA[0]]] for miRNA in self.data.trainingData]
        values = [float(miRNA[2])/sqrt(len(self.data.trainSet_u[miRNA[0]]))/sqrt(len(self.data.trainSet_i[miRNA[1]])) for miRNA in self.data.trainingData]*2
        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.num_circRNAs+self.num_miRNAs,self.num_circRNAs+self.num_miRNAs])
        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        weight_size = [self.emb_size, self.emb_size, self.emb_size,self.emb_size,self.emb_size] #can be changed
        weight_size_list = [self.emb_size] + weight_size
        self.n_layers = 2#layers参数在这里

        #initialize parameters
        for k in range(self.n_layers):
            self.weights['W_%d_1' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_1' % k)
            self.weights['W_%d_2' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_2' % k)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(norm_adj,ego_embeddings)
            sum_embeddings = tf.matmul(side_embeddings+ego_embeddings, self.weights['W_%d_1' % k])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2' % k])
            ego_embeddings = tf.nn.leaky_relu(sum_embeddings+bi_embeddings)
            # message dropout.
            def without_dropout():
                return ego_embeddings
            def dropout():
                return tf.nn.dropout(ego_embeddings, keep_prob=0.9)
            ego_embeddings = tf.cond(self.isTraining,lambda:dropout(),lambda:without_dropout())
            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        all_embeddings1 = tf.concat(all_embeddings, 1)

        all_embeddings2 = tf.reduce_sum(all_embeddings, axis=0)
        self.multi_circRNA_embeddings2, self.multi_miRNA_embeddings2 = tf.split(all_embeddings2,
                                                                          [self.num_circRNAs, self.num_miRNAs], 0)

        self.multi_circRNA_embeddings, self.multi_miRNA_embeddings = tf.split(all_embeddings1, [self.num_circRNAs, self.num_miRNAs], 0)
        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_miRNA_embedding = tf.nn.embedding_lookup(self.multi_miRNA_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.multi_circRNA_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.multi_miRNA_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,self.multi_miRNA_embeddings),1)

    def buildModel(self):
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_miRNA_embedding), 1)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_miRNA_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch_pairwise()):
                circRNA_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: circRNA_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                print('training:', iteration + 1, 'batch', n, 'loss:', l)

        self.U, self.V = self.sess.run([self.multi_circRNA_embeddings2, self.multi_miRNA_embeddings2], feed_dict={self.isTraining:0})
        np.save("circRNA.npy", self.U)
        # np.save('circRNAembedding',c)





        # print('tensorflow中的embedding',type(self.u_embedding),type(self.v_embedding))
        # print(type(self.u_embedding))

        # with tf.Session() as sess:
        #
        #     np.save('x.npy', sess.run(self.u_embedding), allow_pickle=False)

        # 指定会话

            # sess = tf.Session()
            # print(sess.run(self.u_embedding))


            # with tf.Session()  as sess:
            #     print(self.u_embedding.eval())
            # tf.print('tf中的张量',self.u_embedding)
            # with tf.Session() as sess:
            #     print(sess.run(self.u_embedding))
            #     print(sess.run(self.v_embedding))
            # with tf.Session() as sess:
            #     tf.print(sess.run(self.u_embedding))

        # with tf.Session() as sess:
        #     print(sess.run(self.u_embedding))


    def predictForRanking(self, u):
        'invoked to rank all the miRNAs for the circRNA'
        if self.data.containscircRNA(u):
            u = self.data.getcircRNAId(u)
            # print(u)
            return self.sess.run(self.test,feed_dict={self.u_idx:u,self.isTraining:0})
        else:
            return [self.data.globalMean] * self.num_miRNAs
    def predictForRankingmiRNA(self,i):
        if self.data.containsmiRNA(i):
            i=self.data.getmiRNAId(i)
            return self.sess.run(self.test,feed_dict={self.i_idx:i,self.isTraining:0})
        else:
            return [self.data.globalMean]*self.num_circRNAs