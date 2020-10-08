import numpy as np

def dot_product_similarity(gesture1, gesture2):
    assert isinstance(gesture1, list)
    assert isinstance(gesture2, list)

    return np.dot(gesture2, gesture1)