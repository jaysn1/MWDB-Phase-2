import numpy as np
import json

from Phase2 import task1
import task1 as Phase3task1
from lsh import deserialize_vector_file, LSH
from GausianNaiveBayesClassifier import GausianNaiveBayesClassifier, GaussianNaiveBayesClf

class ProbabilityFeedbackNBModel():
    """
    This class provides methods to run relevent feedback system.
    So what is happening here is that we use NaiveBayes method (with the assumption that words are independent), to see if the given gesture is relevent or not.
    By doing so, we reduce our scope to only consider the gestures we find relevent. The scoping is done by our NaiveBayes model as it removes the gestures that 
    of no interest to us. Then we run LSH again on the new set of gestures, and provide the results for the user's query.
    """
    def __init__(self, vector_model, num_layers=4, num_hash_per_layer=8):
        # word position dictionary
        word_pos_dict_dir = "intermediate/word_position_dictionary.json"
        self.word_position_dict = {}
        with open(word_pos_dict_dir) as f:
            for k,v in json.load(f).items():
                k = eval(k)
                self.word_position_dict[v] = tuple([k[0]] + [k[1]] + list(k[2]))

        file_name = "intermediate/vectors_dictionary.json"
        self.vectors = deserialize_vector_file(file_name)
        self.gesture_ids = list(self.vectors.keys())
        self.vector_model = vector_model

        np.random.seed(0)
        self.num_layers = num_layers
        self.num_hash_per_layer = num_hash_per_layer
        self.input_vector_dimension = len(self.vectors[self.gesture_ids[0]][vector_model])
        self.lsh = LSH(self.num_layers, self.num_hash_per_layer, input_dimension=self.input_vector_dimension)
        self.lsh.index(self.vectors)

    def initial_query(self, query_id, t=15):
        self.query = self.vectors[query_id][self.vector_model]
        results = self.lsh.query(point=self.query, t=t, vectors=self.vectors)
        return results

    def make_data(self, relevent_ids, irrevelvent_ids):
        similarity_dict = Phase3task1.task1_initial_setup(2, self.vector_model, create_vectors=False)
        
        train_relevent = []
        for relevent in relevent_ids:
            train_relevent.extend([_[0] for _ in sorted(similarity_dict[relevent].items(), key=lambda x:-x[1])][:15])
        train_relevent.extend(relevent_ids)

        train_irrelevent = []
        for irrelevent in irrevelvent_ids:
            train_irrelevent.extend([_[0] for _ in sorted(similarity_dict[irrelevent].items(), key=lambda x:-x[1])][:15])
        train_irrelevent.extend(irrevelvent_ids)

        train_irrelevent = [i for i in train_irrelevent if i not in train_relevent]

        x_train = [self.vectors[_][self.vector_model] for _ in train_relevent+train_irrelevent]
        y_train = [1 for _ in train_relevent] + [0 for _ in train_irrelevent]

        return x_train, y_train

    def iteration(self, relevent_ids, irrelevent_ids, t=15):
        x_train, y_train = self.make_data(relevent_ids, irrelevent_ids)

        assert len(x_train) == len(y_train)
        # from sklearn.naive_bayes import GaussianNB
        # clf = GaussianNB()
        # clf.fit(np.array(x_train), np.array(y_train))
        # clf = GausianNaiveBayesClassifier(x_train, y_train)
        clf = GaussianNaiveBayesClf(x_train, y_train)
        print("Relative importance highest 10: ")
        rel_imp = [(self.word_position_dict[i],_) for i,_ in sorted(enumerate(clf.imp), key=lambda x :x[1] ,reverse=True)]
        print(rel_imp[:10])
        print("Relative importance lowest 10: ")
        print(rel_imp[-10:])

        y_pred = clf.predict([self.vectors[gesture][self.vector_model] for gesture in self.vectors])
        self.vectors = {gesture: self.vectors[gesture] for i,gesture in enumerate(self.vectors) if y_pred[i] == 1}
        self.gesture_ids = list(self.vectors.keys())

        self.lsh = LSH(self.num_layers, self.num_hash_per_layer, input_dimension=self.input_vector_dimension)
        self.lsh.index(self.vectors)
        
        return self.lsh.query(point=self.query, t=15, vectors=self.vectors)
        

if __name__ == '__main__':
    vector_model = 0
    obj = ProbabilityFeedbackNBModel(vector_model=0, num_layers=4, num_hash_per_layer=5)
    print("\nInitial Results: ", [_[0] for _ in obj.initial_query(input("Query: "))])
    while(True):  # Until user is satisfied or we do not have anymore results to show
        relevent_ids = list(map(lambda x:x.strip(), input("relevent: ").split(",")))
        notrelevent_ids = list(map(lambda x:x.strip(), input("Not-relevent: ").split(",")))
        results = obj.iteration(relevent_ids, notrelevent_ids)
        print("\nResults: ", [_[0] for _ in results])
        