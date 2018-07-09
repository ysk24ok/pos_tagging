import numpy as np

from pos_tagging import structured_perceptron


def test_backward():
    uniq_pos = ['名詞', '動詞']
    model = structured_perceptron.StructuredPerceptron(uniq_pos)
    best_score = np.array([
        [2, 3],
        [7, 10],
        [14, 16],
        [15, 18]
    ])
    backpointer = np.array([
        [-1, -1],
        [0, 1],
        [1, 0],
        [-1, -1],
    ])
    y_pred = model.backward(backpointer, best_score)
    assert len(y_pred) == best_score.shape[0] - 1
