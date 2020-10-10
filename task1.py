from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
from pandas import DataFrame
from functools import wraps
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def pca(model):
    print("pca")


def svd(model):
    print("svd")


def nmf(model):
    print("nmf")


def lda(model):
    print("lda")


def select_method(model):
    print("Select method:")
    print("1. PCA")
    print("2. SVD")
    print("3. NMF")
    print("4. LDA")
    try:
        option = input("Enter option: ")
        method = int(option)
        if method == 1:
            pca(model)
        elif method == 2:
            svd(model)
        elif method == 3:
            nmf(model)
        elif method == 4:
            lda(model)
        else:
            exit(400)
    except ValueError:
        exit(500)


def main():
    print("Select a vector model:")
    print("1. TF")
    print("2. TF-IDF")
    try:
        option = input("Enter option: ")
        model = int(option)
        if model == 1 or model == 2:
            select_method(model)
        else:
            exit(400)
    except ValueError:
        exit(500)


if __name__ == '__main__':
    main()
