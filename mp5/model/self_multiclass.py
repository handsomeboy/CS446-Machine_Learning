import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        ret_dict = {}
        for label in self.labels:
            b_y = np.where(y==label, 1, 0)
            clf = svm.LinearSVC(random_state=12345)
            clf.fit(X,b_y)
            ret_dict[label] = clf

        return ret_dict

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        ret_dict = {}
        X_y = np.hstack([X, y.reshape(-1,1)])
        for i, label in enumerate(self.labels[:-1]):
            for other_label in self.labels[i+1:]:
                b_X_y = X_y[np.where((X_y[:,-1]==label) | (X_y[:,-1]==other_label))]
                b_X = b_X_y[:,:-1]
                b_y = b_X_y[:,-1]
                b_y = np.where(b_y==label, 1, 0)
                clf = svm.LinearSVC(random_state=12345)
                clf.fit(b_X,b_y)
                ret_dict[(label,other_label)] = clf

        return ret_dict

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        ret_matrix = np.zeros((X.shape[0], self.labels.shape[0]))
        for label in self.labels:
            clf = self.binary_svm[label]
            col = clf.decision_function(X)
            ret_matrix[:,label] = col

        return ret_matrix

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        ret_matrix = np.zeros((X.shape[0], self.labels.shape[0]))
        for i, label in enumerate(self.labels[:-1]):
            for other_label in self.labels[i+1:]:
                clf = self.binary_svm[(label,other_label)]
                vote = clf.predict(X)
                vote_other = np.where(vote==0, 1, 0)
                ret_matrix[:,label] += vote
                ret_matrix[:,other_label] += vote_other

        return ret_matrix

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        K,d = W.shape
        N = X.shape[0]
        sum_w_norm = (W ** 2).sum()/2
        one_minus_delta = np.ones((K, N))
        one_minus_delta[y, np.arange(N)] = 0
        max_score = (np.dot(W,X.T) + one_minus_delta).max(axis=0)
        y_one_hot = np.zeros((N,K))
        y_one_hot[np.arange(N),y] = 1
        W_y = np.dot(y_one_hot, W)
        corr_score = np.diag(np.dot(W_y, X.T))
        loss = sum_w_norm + (max_score - corr_score).sum() * C

        return loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        K,d = W.shape
        N = X.shape[0]
        one_minus_delta = np.ones((K, N))
        one_minus_delta[y, np.arange(N)] = 0
        y_max = np.argmax(np.dot(W,X.T) + one_minus_delta, axis=0)
        max_one_hot = np.zeros((K,N))
        max_one_hot[y_max, np.arange(N)] = 1
        grad_w_j = np.dot(max_one_hot, X)
        y_one_hot = np.zeros((K,N))
        y_one_hot[y, np.arange(N)] = 1
        grad_w_yi = np.dot(y_one_hot, X)
        grad = W + C * (grad_w_j - grad_w_yi)
        
        return grad
