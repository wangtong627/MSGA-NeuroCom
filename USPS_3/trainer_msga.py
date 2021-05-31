import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msga import GATE
from sklearn.preprocessing import OneHotEncoder
from utils import process


class Trainer():

    def __init__(self, args):

        self.args = args
        self.build_placeholders()
        self.gate = GATE(args.hidden_dims, args.lambda_, args.batch_size)
        self.total_loss, self.H, self.C, self.z, self.Coef_mean, self.Clustering_results, self.features_loss \
            , self.structure_loss, self.reg_ssc_loss, self.cost_ssc_loss \
            , self.L_self_supervised_loss = self.gate(self.A, self.X, self.R, self.S)
        self.optimize(self.total_loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, X, S, R, Y):
        list_acc_dsc = []
        list_nmi_dsc = []
        list_ari_dsc = []
        list_loss = []

        for epoch in range(self.args.n_epochs):
            loss, acc_dsc, nmi_dsc, ari_dsc = self.run_epoch(epoch, A, X, S, R, Y)
            list_acc_dsc.append(acc_dsc)
            list_ari_dsc.append(ari_dsc)
            list_nmi_dsc.append(nmi_dsc)
            list_loss.append(loss)

        list_log_dsc = [list_acc_dsc, list_ari_dsc, list_nmi_dsc, list_loss]
        list_log_dsc = np.array(list_log_dsc)
        dataframe_log_dsc = pd.DataFrame(list_log_dsc)
        dataframe_log_dsc.to_csv("usps3_self_supervised_log.csv")

        x = np.arange(self.args.n_epochs)
        plt.plot(x, list_acc_dsc, label='acc_usps')
        plt.plot(x, list_nmi_dsc, label='nmi_usps')
        plt.plot(x, list_ari_dsc, label='ari_usps')
        # plt.plot(x, list_loss, label='list_loss')
        plt.legend(['acc_usps', 'nmi_usps', 'ari_usps'])
        plt.title("usps3_self_supervised")
        plt.show()

        x = np.arange(self.args.n_epochs)
        plt.plot(x, list_loss, label='total_loss')
        plt.legend(['total_loss'])
        plt.title("usps3_self_supervised")
        plt.show()

    def run_epoch(self, epoch, A, X, S, R, Y):

        coef_mean = self.session.run(self.Coef_mean,
                                     feed_dict={self.A: A,
                                                self.X: X,
                                                self.S: S,
                                                self.R: R,
                                                # self.p: p
                                                })
        alpha = max(0.4 - (10 - 1) / 10 * 0.1, 0.1)
        commonZ = self.gate.thrC(coef_mean, alpha)
        pred_dsc, _ = self.gate.post_proC(commonZ, 10)
        _, nmi_dsc, ari_dsc = self.gate.cluster_metrics(Y, pred_dsc)
        acc_dsc = self.gate.cluster_acc(Y, pred_dsc)

        # 将谱聚类的结果转换成0ne-hot 传入自回归计算loss
        pred_dsc_encoded = np.zeros((9298, 10))
        for i in range(9298):
            pred_dsc_encoded[i][pred_dsc[i]] = 1

        total_loss, structure_loss, features_loss, reg_ssc_loss, cost_ssc_loss, l_self_supervised_loss, _ \
            = self.session.run([self.total_loss, self.structure_loss, self.features_loss, self.reg_ssc_loss
                                   , self.cost_ssc_loss, self.L_self_supervised_loss, self.train_op],
                               feed_dict={self.A: A,
                                          self.X: X,
                                          self.S: S,
                                          self.R: R,
                                          # self.p: p
                                          self.Clustering_results: pred_dsc_encoded
                                          })

        print("epoch: {}\ttotal_loss: {}\tacc_dsc: {}\tnmi_dsc: {}\tari_dsc: {}".format(
            epoch, total_loss, acc_dsc, nmi_dsc, ari_dsc))
        print("features: {}\tstructure: {}\treg_c: {}\tcost_c: {}\tself_supervised: {}\n"
            .format(features_loss, structure_loss, reg_ssc_loss, cost_ssc_loss, l_self_supervised_loss))

        return total_loss, acc_dsc, nmi_dsc, ari_dsc

