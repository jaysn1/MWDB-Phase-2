import numpy as np
from scipy.stats import multivariate_normal as mvn
class GausianNaiveBayesClassifier:
    def __init__(self, data, predictions):
        # data will be vectors (2D)
        # prediction will be vector with same rows as data
        self.X = np.array(data)
        self.Y = np.array(predictions)

        self.class_labels = list(set(self.Y))
        self.prioir = {}
        self.class_data = {c:[] for c in self.class_labels}
        for (x,y) in zip(self.X, self.Y):
            self.class_data[y].append(x)
            
        self.class_mean = {}
        self.class_var = {}
        for c in self.class_labels:
            mean, var = self.get_mean_var(self.class_data[c])
            self.class_mean[c] = mean
            self.class_var[c] = var
            # self.pdf = 
            self.prioir[c] = len(self.class_data[c])/len(self.X)

    # Get mean and variance of each image
    def get_mean_var(self, X):
        mean = np.mean(X, axis=0) #[np.mean(np.mean(X , axis = 0)), np.mean(np.var(X , axis = 0))]
        var = np.var(X, axis=0) #[np.var(np.mean(X , axis = 0)), np.var(np.var(X , axis = 0))]
        return mean, var

    # Predicts the probability of an image being in a class using the multivariate formula.
    def predict(self, X):
        rows = len(X)
        columns = len(self.class_labels)
        pred_prob = np.zeros((rows,columns))
        predictions = []
        for i in range(len(X)):
            for j in range(columns):
                c = self.class_labels[j]
                pred_prob[i,j] = (mvn.logpdf(X[i], mean=self.class_mean[c], cov=self.class_var[c], allow_singular=True) + np.log(self.prioir[c]))
            predictions.append(self.class_labels[np.argmax(pred_prob[i,:])])
        return predictions

# ref: https://github.com/ScienceKot/mysklearn/blob/master/naive_bayes/gauss.py
import math
import numpy as np
class GaussianNaiveBayesClf:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.fit()

        self.imp = [0 for _ in range(self.X.shape[1])]    
        means0, stds0 = self.means[0], self.std[0]
        means1, stds1 = self.means[1], self.std[1]
        # finding intersection point
        for i in range(self.X.shape[1]):
            if stds0[i]==0 or stds1[i]==0:
                continue
            else:
                # point = self.solve(means0[i], means1[i], stds0[i], stds1[i])
                area_0, area_1 = self.calculate_area(means0[i], means1[i], stds0[i], stds1[i], i)
                if means0[i] < means1[i]:
                    self.imp[i] = area_1 / (area_0 + area_1)
                else:
                    self.imp[i] = area_0 / (area_0 + area_1)

    def fit(self):
        ''' The fitting function '''
        X, y = self.X, self.y

        self.classes = np.unique(y)
        classes_index = {}
        separated_X = {}
        cls, counts = np.unique(y, return_counts=True)
        self.class_freq = dict(zip(cls, counts))
        # print(self.class_freq)
        s = sum(list(self.class_freq.values()))
        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(y==class_type)
            separated_X[class_type] = X[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type]/s

        self.means = {}
        self.std = {}
        for class_type in self.classes:
            # Here we calculate the mean and the standart deviation from datasets
            self.means[class_type] = np.mean(separated_X[class_type], axis=0)[0]
            self.std[class_type] = np.std(separated_X[class_type], axis=0)[0]

    def calculate_area(self, mean0, mean1, std0, std1, ind):
        if mean1<mean0:
            mean0, mean1 = mean1, mean0
            std1, std0 = std0, std1

        class_prob = {cls:math.log(self.class_freq[cls], math.e) for cls in self.classes}
        a0, a1 = 0,0
        for i in range(self.X.shape[0]):
            class_prob0 = class_prob[0] + math.log(self.calculate_probability(self.X[i, ind], mean0, std0, 0), math.e)
            class_prob1 = class_prob[1] + math.log(self.calculate_probability(self.X[i, ind], mean1, std1, 0), math.e)

            if class_prob0 > class_prob1:
                a0 += 1
            else:
                a1 += 1

        return a0, a1


    def calculate_probability(self, x, mean, stdev, cls):
        ''' This function calculates the class probability assuming a gaussian distribution '''
        if stdev == 0:
            return math.e
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))    
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict_proba(self, X):
        ''' This function predicts the probability for every class '''
        self.class_prob = {cls:math.log(self.class_freq[cls], math.e) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(self.means)):
                self.class_prob[cls] += math.log(self.calculate_probability(X[i], self.means[cls][i], self.std[cls][i], cls), math.e)
        self.class_prob = {cls: math.e**self.class_prob[cls] for cls in self.class_prob}
        return self.class_prob

    def predict(self, X):
        ''' This funtion predicts the class of a sample '''
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for cls, prob in self.predict_proba(x).items():
                if prob>max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred




if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=5, centers=2, n_features=2, random_state=1)
    # print(y)
    # obj = GausianNaiveBayesClassifier(X, y)
    # print(obj.predict(X))
    print(y)
    obj = GaussianNaiveBayesClf(X, y)
    pred = obj.predict(X)
    print(pred)
    print(obj.imp)