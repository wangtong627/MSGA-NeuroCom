import argparse

import numpy as np
# from utils.classifier import Classifier
from trainer_msga import Trainer
from utils import process
from warnings import simplefilter
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run gate.")

    # parser.add_argument('--dataset', nargs='?', default='citeseer',
    #                     help='Input dataset')

    parser.add_argument('--lr', type=float, default=1.0e-4,
                        help='Learning rate. Default is 0.001/3e-5')

    parser.add_argument('--n-epochs', default=50, type=int,
                         help='Number of epochs')

    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[512, 512],
                        help='Number of dimensions.')

    parser.add_argument('--lambda-', default=1.0, type=float,
                        help='Parameter controlling the contribution of edge reconstruction in the loss function.')

    parser.add_argument('--dropout', default=0.0, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    # parser.add_argument('--batch_size', default=3327, type=int,
    #                     help='batch_size')

    return parser.parse_args()


def main(args):

    '''
    datasets preprocessing
    '''

    data_0 = sio.loadmat('Dataset/hhar_5.mat')
    data_dict = dict(data_0)
    G = sp.coo_matrix(data_dict['adj'])
    X = np.matrix(data_dict['x'])
    Y = np.squeeze(data_dict['y'])-1

    Y_encoded = np.array(Y).reshape(len(Y), -1)
    enc = OneHotEncoder()
    enc.fit(Y_encoded)
    Y = enc.transform(Y_encoded).toarray()

    # add feature dimension size to the beginning of hidden_dims
    feature_dim = X.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims
    args.batch_size = X.shape[0]

    # prepare the data
    G_tf, S, R = process.prepare_graph_data(G)

    # Train the Model
    trainer = Trainer(args)
    trainer(G_tf, X, S, R, Y)
    # embeddings, attentions = trainer.infer(G_tf, X, S, R)

    # Evaluate the quality of embeddings
    # classifier = Classifier(vectors=embeddings)
    # f1s = classifier(idx_train, idx_test, idx_val, Y, seed=0)
    # print(f1s)


if __name__ == "__main__":
    args = parse_args()
    main(args)
