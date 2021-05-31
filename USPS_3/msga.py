import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn import cluster
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class GATE():

    def __init__(self, hidden_dims, lambda_, batch_size):
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size

    def __call__(self, A, X, R, S):

        # 自表达的参数定义部分
        Coef_0 = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_0')
        Coef_1 = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_1')
        Coef_2 = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef_2')

        weight_0 = tf.Variable(1.0, name='weight_0')
        weight_1 = tf.Variable(1.0, name='weight_1')
        weight_2 = tf.Variable(1.0, name='weight_2')

        self.Coef = [Coef_0, Coef_1, Coef_2]
        self.z_ssc = []
        cost_ssc = 0
        reg_ssc = 0

        # Encoder
        self.X = X
        self.H_in = []
        H = X
        self.H_in.append(H)
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)
            self.H_in.append(H)

        # Final node representations
        self.H = H

        self.z = self.H

        for i in range(self.n_layers + 1):
            z_ssc = tf.matmul(self.Coef[i], self.H_in[i])
            self.z_ssc.append(z_ssc)

        self.Coef_mean = 1 / (weight_0 + weight_1 + weight_2) * (weight_0 * Coef_0 + weight_1 * Coef_1 + weight_2 * Coef_2)
        # self.Coef_mean = 1 / 3 * (Coef_0 + Coef_1 + Coef_2)

        # Decoder_1
        self.H_out_1 = []
        H = tf.matmul(self.Coef_mean, self.z)
        self.H_out_1.append(H)
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
            self.H_out_1.append(H)
        self.H_out_1.reverse()
        X_1 = H

        # 自监督
        Pseudo_L = self.__dense_layer(self.z)
        self.Clustering_results = tf.placeholder(dtype=tf.float32)
        L_self_supervised = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Pseudo_L, logits=self.Clustering_results))

        # 这边计算loss
        # decoder1 loss, The reconstruction loss of node features
        features_loss_1 = 1/2 * (tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_1, 2)))) \
                          # + 1.0 * tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(self.H_in[1] - self.H_out_1[1], 2))))


        # 自表达的loss部分 6个
        for i in range(self.n_layers + 1):
            cost_ssc = 1/2 * (tf.reduce_sum(tf.pow(tf.subtract(self.H_in[i], self.z_ssc[i]), 2)))
            reg_ssc = tf.reduce_mean(tf.pow(self.Coef[i], 2)) # reduce_sum ?
            cost_ssc += cost_ssc
            reg_ssc += reg_ssc
        self.reg_ssc = 0.334 * reg_ssc
        self.cost_ssc = 0.334 * cost_ssc

        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss = 1.0 * (tf.reduce_sum(structure_loss))

        # MSE loss
        # with tf.name_scope("distribution"):
        #     self.q = self._soft_assignment(self.z, self.mu)
        #     self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))
        #     self.pred = tf.argmax(self.q, axis=1)
        # with tf.name_scope("MSE_loss"):
        #     self.MSE_loss = self._MSE(self.p, self.q)
        # # kl
        # with tf.name_scope("KL"):
        #     self.KL_divergence_loss = self._KL_divergence(self.p, self.q)

        self.features_loss = 1.0 * features_loss_1
        self.structure_loss = 1.0 * structure_loss  # self.lambda_ = 0.1
        self.reg_ssc_loss = 1.0 * self.reg_ssc
        self.cost_ssc_loss = 1.0 * self.cost_ssc
        self.L_self_supervised_loss = 1.0 * L_self_supervised

        # Total loss
        self.total_loss = self.features_loss + self.structure_loss + self.reg_ssc_loss \
                          + self.cost_ssc_loss + self.L_self_supervised_loss

        return self.total_loss, self.H, self.C, self.z, self.Coef_mean, self.Clustering_results \
            , self.features_loss, self.structure_loss, self.reg_ssc_loss, self.cost_ssc_loss \
            , self.L_self_supervised_loss


    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i + 1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i + 1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def graph_attention_layer(self, A, M, v, layer):

        with tf.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                                      values=tf.nn.sigmoid(logits.values),
                                                      dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions

    def _MSE(self, target, pred):
        return tf.reduce_mean((target - pred) ** 2)

    def _KL_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))

    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        y_true = np.argmax(y_true, axis=1)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def cluster_metrics(self, y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_true = np.argmax(y_true, axis=1)
        acc = accuracy_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        return acc, nmi, ari

    def post_proC(self, C, K, d=10, alpha=8):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L)
        return grp, L

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while stop == False:
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C

        return Cp

    def __dense_layer(self, Z):
        dense1 = tf.layers.dense(inputs=Z, units=128, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
        dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense3, units=10, activation=None)
        return logits
