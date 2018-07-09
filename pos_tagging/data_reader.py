def readline(line):
    """
    Args:
        line (str): one line of a data file
    Returns:
        list[str]: word sequence of one sentence
        list[str]: pos sequence of one sentence
    """
    x = []
    y = []
    for word_pos in line.strip().split(' '):
        word, pos = word_pos.split('_')
        x.append(word)
        y.append(pos)
    return x, y


def read_data(filepath):
    """
    Args:
        filepath: str
    Returns:
        list[list[str]]: word sequence of multiple sentences
        list[list[str]]: pos sequence of multiple sentences
    """
    X = []
    Y = []
    with open(filepath, 'r') as f:
        for line in f:
            x, y = readline(line)
            X.append(x)
            Y.append(y)
    return X, Y


def unique_pos(Y):
    """
    Args:
        Y (list[list[str]]): pos sequence of multiple sentences
    Returns:
        list[str]: unique pos
    """
    unique_pos = set([])
    for y in Y:
        unique_pos = unique_pos ^ set(y)
    return list(unique_pos)
