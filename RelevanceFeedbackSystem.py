import numpy as np
import json

from Phase2 import task1
from lsh import deserialize_vector_file, LSH
from Phase1 import helper

"""
This class is for implmentation of relevence feedback system using vector space model as well as probabilistic method. 
Assumption: For probabilistic model, we assume a Binary Independence Model, so vector representation are converted to binary
"""
class PrababilityFeedbackModel():
    def __init__(self, vector_model, num_layers=4, num_hash_per_layer=8):
        # word position dictionary
        word_pos_dict_dir = "intermediate/word_position_dictionary.json"
        self.word_position_dict = {}
        with open(word_pos_dict_dir) as f:
            for k,v in json.load(f).items():
                self.word_position_dict[ v ] = eval(k)

        file_name = "intermediate/vectors_dictionary.json"
        self.vectors = deserialize_vector_file(file_name)
        self.gesture_ids = list(self.vectors.keys())
        self.documents = np.array([[1 if _>0 else 0 for _ in self.vectors[gesture][vector_model] ] for gesture in self.gesture_ids])
        self.vector_model = vector_model

        np.random.seed(0)
        input_vector_dimension = len(self.vectors[self.gesture_ids[0]][vector_model])
        self.lsh = LSH(num_layers, num_hash_per_layer, input_dimension=input_vector_dimension)
        self.lsh.index(self.vectors)

        self.relevent_ids = set()
        self.irrelevent_ids = set()



    def initial_query(self, query_id, t=15):
        self.query = self.vectors[query_id][self.vector_model]
        # print("Query: ", [(i,_) for i,_ in enumerate(self.query) if _!=0 ][:10])
        results = self.lsh.query(point=self.query, t=t, vectors=self.vectors)

        query_document = np.array([1 if _>0 else 0 for _ in self.query])
        documents = self.documents
        self.r1 = [0.5 for i in range(len(query_document))]
        self.n1 = np.divide(np.sum(documents, axis=0), len(documents))

        self.n2 = [0.5 for i in range(len(query_document))]
        self.r2 = np.divide(np.sum(documents, axis=0), len(documents))

        return results

    def iteration(self, relevent_gesture_ids, irrelevent_gesture_ids, t=15):
        # self.relevent_ids.update(relevent_gesture_ids)
        # self.irrelevent_ids.update(irrelevent_gesture_ids)
        # self.irrelevent_ids = self.irrelevent_ids.difference(self.relevent_ids)
        
        # print(len(self.relevent_ids), len(self.irrelevent_ids))

        relevent_ids = relevent_gesture_ids # self.relevent_ids
        irrelevent_ids = irrelevent_gesture_ids #self.irrelevent_ids

        L = len(relevent_ids)
        K = len(irrelevent_ids) + len(relevent_ids)

        query_document = np.array([1 if _>0 else 0 for _ in self.query])
        relevent_documents = np.array([[1 if _>0 else 0 for _ in self.vectors[gesture][vector_model]] for gesture in relevent_ids])
        
        l = relevent_documents.sum(axis=0)
        k = np.divide(l, len(relevent_documents))    
        for i in range(len(query_document)):
            if query_document[i] != 0:
                self.r1[i] = (l[i] + 0.5) / (L + 1)
                self.n1[i] = (k[i]-l[i]+0.5 )/(K+L+1)

        L2 = len(irrelevent_ids) + len(relevent_ids)
        K2 = len(relevent_ids)

        
        irrelevent_documents = np.array([[1 if _>0 else 0 for _ in self.vectors[gesture][vector_model]] for gesture in irrelevent_ids])
        
        l = irrelevent_documents.sum(axis=0)
        k = np.divide(l, len(irrelevent_documents))    
        for i in range(len(query_document)):
            if query_document[i] != 0:
                self.r2[i] = (l[i] + 0.5) / (L2 + 1)
                self.n2[i] = (k[i]-l[i]+0.5 )/(K2+L2+1)


        weight = []
        for i in range(len(query_document)):
            if (self.n1[i])==0:
                p1 = 1
            else:
                p1 = self.r1[i]*(1-self.n1[i]) / (self.n1[i] * (1-self.r1[i]))

            if (self.n2[i])==0:
                p2 = 1
            else:
                p2 = self.r2[i]*(1-self.n2[i]) / (self.n2[i] * (1-self.r2[i]))
            
            if p1>p2:
                weight.append(p1)
            else:
                weight.append(-p1)

        sw = len(weight)
        updated_query = [self.query[i]*_ for i,_ in enumerate(weight)]
        # weight = helper.min_max_scaler([weight], feature_range=(min(self.query), max(self.query)))[0]
        # print("Query: ", [(i,_) for i,_ in enumerate(self.query) if _!=0 ][:10])
        # print("Updated query: ", [(i,_) for i,_ in enumerate(weight) if _!=0 ][:10])
        # self.query = weight
        results = self.lsh.query(point=updated_query, t=t, vectors=self.vectors)
        return results

if __name__ == '__main__':
    vector_model = 0
    obj = PrababilityFeedbackModel(vector_model=0, num_layers=4, num_hash_per_layer=5)
    print("\ninitial Results: ", [_[0] for _ in obj.initial_query(input("Query: "))])
    
    for i in range(4):
        relevent_ids = list(map(lambda x:x.strip(), input("relevent: ").split(",")))
        notrelevent_ids = list(map(lambda x:x.strip(), input("Not-relevent: ").split(",")))
        results = obj.iteration(relevent_ids, notrelevent_ids)
        print("\nReuslts: ", [_[0] for _ in results])
        