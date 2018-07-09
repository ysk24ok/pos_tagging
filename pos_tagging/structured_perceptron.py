import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import shuffle

from . import data_reader


def set_feature(f, idx, current_pos, prev_pos, x):
    """
    Args:
        f (dict[str, int]): feature mapping
        idx (int): Current index in a sentence
        current_pos (str): POS of the current position
        prev_pos (str): POS of the previous position
        x (list[str]): Word sequence of one sentence
    """
    # TODO: current_pos='EOS'のとき、x[idx]はindex errorになる
    # 品詞Nのあとに品詞Nがつづく
    if 'N_followed_by_N' not in f:
        f['N_followed_by_N'] = 0
    if current_pos == '名詞' and prev_pos == '名詞':
        f['N_followed_by_N'] += 1
    # 品詞Nのあとに品詞Vがつづく
    if 'N_followed_by_V' not in f:
        f['N_followed_by_V'] = 0
    if current_pos == '名詞' and prev_pos == '動詞':
        f['N_followed_by_V'] += 1
    # 品詞Vのあとに品詞Nがつづく
    if 'V_followed_by_N' not in f:
        f['V_followed_by_N'] = 0
    if current_pos == '動詞' and prev_pos == '名詞':
        f['V_followed_by_N'] += 1
    # BOSのあとに品詞Nがつづく
    if 'BOS_followed_by_N' not in f:
        f['BOS_followed_by_N'] = 0
    if current_pos == '名詞' and prev_pos == 'BOS':
        f['BOS_followed_by_N'] += 1
    # 品詞Nのあとに助詞がつづく
    if 'N_followed_by_J' not in f:
        f['N_followed_by_J'] = 0
    if current_pos == '助詞' and prev_pos == '名詞':
        f['N_followed_by_J'] += 1
    # 連体詞のあとに名詞がつづく
    if 'R_followed_by_N' not in f:
        f['R_followed_by_N'] = 0
    if current_pos == '名詞' and prev_pos == '連体詞':
        f['R_followed_by_N'] += 1
    # 形容詞のあとに語尾がつづく
    if 'ADJ_followed_by_Gobi' not in f:
        f['ADJ_followed_by_Gobi'] = 0
    if current_pos == '語尾' and prev_pos == '形容詞':
        f['ADJ_followed_by_Gobi'] += 1
    # 形状詞のあとに助動詞がつづく
    if 'KJ_followed_by_JD' not in f:
        f['KJ_followed_by_JD'] = 0
    if current_pos == '助動詞' and prev_pos == '形状詞':
        f['KJ_followed_by_JD'] += 1
    # 接頭辞のあとに名詞がつづく
    if 'Prefix_followed_by_N' not in f:
        f['Prefix_followed_by_N'] = 0
    if current_pos == '名詞' and prev_pos == '接頭辞':
        f['Prefix_followed_by_N'] += 1
    # 名詞のあとに接尾辞がつづく
    if 'N_followed_by_Suffix' not in f:
        f['N_followed_by_Suffix'] = 0
    if current_pos == '接尾辞' and prev_pos == '名詞':
        f['N_followed_by_Suffix'] += 1


def accuracy(X, Y, Y_pred):
    total = 0
    correct = 0
    for x, y, y_pred in zip(X, Y, Y_pred):
        for y_pos, y_pred_pos in zip(y, y_pred):
            total += 1
            if y_pos == y_pred_pos:
                correct += 1
    return correct / total


class StructuredPerceptron(object):

    def __init__(self, uniq_pos, num_epochs=3, batch_size=1):
        self.uniq_pos = uniq_pos
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dict_vectrizer = DictVectorizer(sparse=False)
        self.weights = None

    def init_weights(self, X, Y):
        """Initialize DictVectorizer and weight vector
        Args:
            X (list[list[str]]): word sequence of multiple sentences
            Y (list[list[str]]): pos sequence of multiple sentences
        """
        x = X[0]
        f = {}
        set_feature(f, 0, self.uniq_pos[0], 'BOS', x)
        v = self.dict_vectrizer.fit_transform(f)
        self.weights = np.random.randn(*v.shape) / 100

    def backward(self, edge_backpointer, best_score):
        """Trace edge back pointer to find a best tag sequence
        Args:
            edge_backpointer (2D numpy aray)
            best score (2D numpy array)
        Returns:
            (list[str]): Predicted pos sequence of one sentence
        """
        y_pred = []
        pos_idx = np.argmax(best_score[-1])
        y_pred.insert(0, self.uniq_pos[pos_idx])
        word_idx = edge_backpointer.shape[0] - 1
        while word_idx >= 0:
            prev_pos_idx = edge_backpointer[word_idx][pos_idx]
            word_idx -= 1
            if prev_pos_idx == -1:
                continue
            y_pred.insert(0, self.uniq_pos[prev_pos_idx])
            pos_idx = prev_pos_idx
        return y_pred

    def edge_score(self, word_idx, cur_pos, prev_pos, x):
        f = {}
        set_feature(f, word_idx, cur_pos, prev_pos, x)
        v = self.dict_vectrizer.transform(f)
        return (self.weights @ v.T)[0]

    def viterbi(self, x):
        """
                        word_idx  prev_pos  cur_pos
              BOS
            /  |  \        0         BOS      any
           N   V   ADJ
           | X | X |       1         any      any
           N   V   ADJ
           | X | X |       2         any      any
           N   V   ADJ
            \  |  /        3         any      EOS
              EOS
        """
        best_score = np.full((len(x)+1, len(self.uniq_pos)), -np.inf)
        edge_backpointer = np.full((len(x)+1, len(self.uniq_pos)), -1)
        # First edge
        word_idx = 0
        for cur_pos_idx in range(len(self.uniq_pos)):
            cur_pos = self.uniq_pos[cur_pos_idx]
            score = self.edge_score(word_idx, cur_pos, 'BOS', x)
            best_score[word_idx][cur_pos_idx] = score
            edge_backpointer[word_idx][cur_pos_idx] = -1
        # Loop
        for word_idx in range(1, len(x)):
            for cur_pos_idx in range(len(self.uniq_pos)):
                cur_pos = self.uniq_pos[cur_pos_idx]
                for prev_pos_idx in range(len(self.uniq_pos)):
                    prev_pos = self.uniq_pos[prev_pos_idx]
                    prev_score = best_score[word_idx-1][prev_pos_idx]
                    score = self.edge_score(word_idx, cur_pos, prev_pos, x)
                    score += prev_score
                    if score > best_score[word_idx][cur_pos_idx]:
                        best_score[word_idx][cur_pos_idx] = score
                        edge_backpointer[word_idx][cur_pos_idx] = prev_pos_idx
        # Last edge
        word_idx = len(x)
        for cur_pos_idx in range(len(self.uniq_pos)):
            cur_pos = self.uniq_pos[cur_pos_idx]
            prev_score = best_score[word_idx-1][cur_pos_idx]
            score = self.edge_score(word_idx, 'EOS', cur_pos, x)
            best_score[word_idx][cur_pos_idx] = score + prev_score
            edge_backpointer[word_idx][cur_pos_idx] = -1
        # Trace edge backpointer
        return self.backward(edge_backpointer, best_score)

    def predict(self, X):
        """Forward algorithm, argmax
        Args:
            X (list[list[str]]): Word sequence of multiple sentences
        Returns:
            (list[list[str]]): Predicted pos sequence of multiple sentences
        """
        Y_pred = []
        for x in X:
            y_pred = self.viterbi(x)
            Y_pred.append(y_pred)
        return Y_pred

    def get_feature_vector(self, x, y):
        """Get f(x, y), a feature vector of one sentence
        Args:
            x (list[str]): Word sequence of one sentence
            y (list[str]): POS sequence of one sentence
        Returns:
            (2D numpy array): Feature vector of one sentence
        """
        prev_pos = 'BOS'
        f = {}
        for idx in range(len(x)):
            pos = y[idx]
            set_feature(f, idx, pos, prev_pos, x)
            prev_pos = pos
        set_feature(f, idx+1, 'EOS', prev_pos, x)
        return self.dict_vectrizer.transform(f)

    def update(self, X, Y, Y_pred):
        """Update weights when Y[i] and Y_pred[i] are not equal
        Args:
            X (list[list[str]]): Word sequences of multiple sequences
            Y (list[list[str]]): POS sequences of multiple sentences
            Y_pred (list[list[str]]):
                Predicted POS sequences of multiple sentences
        """
        for x, y, y_pred in zip(X, Y, Y_pred):
            if np.all(y == y_pred) is True:
                continue
            v = self.get_feature_vector(x, y)
            v_pred = self.get_feature_vector(x, y_pred)
            self.weights += v - v_pred

    def fit(self, X, Y):
        """Begin training
        Args:
            X (list[list[str]]): Word sequences of multiple sequences
            Y (list[list[str]]): POS sequences of multiple sentences
        """
        self.init_weights(X, Y)
        for cur_epoch in range(self.num_epochs):
            print('current epoch: {}'.format(cur_epoch))
            # X, Y = shuffle(X, Y, random_state=2)
            X, Y = shuffle(X, Y)
            num_iters = 0
            while True:
                p_idx = self.batch_size * num_iters
                n_idx = self.batch_size * (num_iters + 1)
                X_minibatch = X[p_idx:n_idx]
                Y_minibatch = Y[p_idx:n_idx]
                if len(X_minibatch) == 0:
                    break
                num_iters += 1
                Y_pred = self.predict(X_minibatch)
                self.update(X_minibatch, Y_minibatch, Y_pred)


if __name__ == '__main__':
    import sys
    nlptutorial_path = sys.argv[1]
    tr_data_path = '{}/data/wiki-ja-train.word_pos'.format(nlptutorial_path)
    te_data_path = '{}/data/wiki-ja-test.word_pos'.format(nlptutorial_path)
    tr_X, tr_Y = data_reader.read_data(tr_data_path)
    te_X, te_Y = data_reader.read_data(te_data_path)
    uniq_pos = data_reader.unique_pos(tr_Y)
    print('POS: {}'.format(','.join(uniq_pos)))
    # training
    model = StructuredPerceptron(uniq_pos)
    model.fit(tr_X, tr_Y)
    # evaluate
    te_Y_pred = model.predict(te_X)
    print('accuracy: {:.2f} %'.format(accuracy(te_X, te_Y, te_Y_pred) * 100))
    for fname, w in zip(model.dict_vectrizer.feature_names_, model.weights[0]):
        print('{0:>20}: {1:.2f}'.format(fname, w))
