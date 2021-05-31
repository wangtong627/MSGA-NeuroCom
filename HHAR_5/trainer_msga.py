import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msga import GATE


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
        dataframe_log_dsc.to_csv("dataframe_log_dsc.csv")

        x = np.arange(self.args.n_epochs)
        plt.plot(x, list_acc_dsc, label='acc_dsc')
        plt.plot(x, list_nmi_dsc, label='nmi_dsc')
        plt.plot(x, list_ari_dsc, label='ari_dsc')
        # plt.plot(x, list_loss, label='list_loss')
        plt.legend(['acc_dsc', 'nmi_dsc', 'ari_dsc'])
        plt.title("wiki_self_supervised")
        plt.show()

        x = np.arange(self.args.n_epochs)
        plt.plot(x, list_loss, label='total_loss')
        plt.legend(['total_loss'])
        plt.title("HHAR_self_supervised")
        plt.show()

    # def to_categorical(self, y, num_classes=None):
    #     y = np.array(y, dtype='int')
    #     input_shape = y.shape
    #     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    #         input_shape = tuple(input_shape[:-1])
    #     y = y.ravel()
    #     if not num_classes:
    #         num_classes = np.max(y) + 1
    #     n = y.shape[0]
    #     categorical = np.zeros((n, num_classes))
    #     categorical[np.arange(n), y] = 1
    #     output_shape = input_shape + (num_classes,)
    #     categorical = np.reshape(categorical, output_shape)
    #     return categorical

    def run_epoch(self, epoch, A, X, S, R, Y):

        coef_mean = self.session.run(self.Coef_mean,
                                     feed_dict={self.A: A,
                                                self.X: X,
                                                self.S: S,
                                                self.R: R,
                                                # self.p: p
                                                })
        alpha = max(0.4 - (6 - 1) / 10 * 0.1, 0.1)
        commonZ = self.gate.thrC(coef_mean, alpha)
        pred_dsc, _ = self.gate.post_proC(commonZ, 6)
        _, nmi_dsc, ari_dsc = self.gate.cluster_metrics(Y, pred_dsc)
        acc_dsc = self.gate.cluster_acc(Y, pred_dsc)

        # 将谱聚类的结果转换成0ne-hot 传入自回归计算loss
        # print(pred_dsc)
        # pred_dsc_encoded = np.array(pred_dsc).reshape(len(pred_dsc), -1)
        # enc = OneHotEncoder()
        # enc.fit(pred_dsc_encoded)
        # pred_dsc_one_hot_encoded = enc.transform(pred_dsc_encoded).toarray()
        pred_dsc_encoded = np.zeros((10299, 6))
        for i in range(10299):
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

