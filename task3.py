"""
Phase3 task 3
• Task 4: Probabilistic relevance feedback: Implement a probabilistic relevance feedback system to improve nearest
neighbor matches, which enables the user to label some of the results returned by the search task as relevant or irrelevant
and then returns a new set of ranked results, either by revising the query or by re-ordering the existing results. You can
consider,
– the relative importances of sensors,
– the relative importances of the X,Y,Z, and W components,
– the relative importances of the different words.

author: Monil
"""

# https://github.com/laurabalasso/Binary-Independence-Model-

from Phase2 import task1
import numpy as np
from lsh import deserialize_vector_file, LSH
import json

def relevence_feedback_vector_space(relevent_gestures, not_relevent_gestures, query):
    # vector space based relevent feedback
    relevent_gestures_np = np.array([vectors[_][vector_model] for _ in relevent_gestures])
    centroid_of_relevent = relevent_gestures_np.sum(axis=0) / len(relevent_gestures)

    not_relevent_gestures_np = np.array([vectors[_][vector_model] for _ in not_relevent_gestures])
    centroid_of_not_relevent = not_relevent_gestures_np.sum(axis=0) / len(not_relevent_gestures)

    query = query + centroid_of_relevent - centroid_of_not_relevent

    return query
    ########################################################################

def relevence_feedback_vector_space(relevent_gestures, not_relevent_gestures, query):
    pass


def main():
    """
    This function uses gesture representation from task1 in phase 2 as source for gestures.
    It will ask for a query gesture initially and show the results from nearest neigbor (LSH).
    Then, it will ask user to provide some relevent and non-revelent results from it,
    and it will refine query using "Rocchio algorithm" to create probabilistic feedback
    """
    # word position dictionary
    word_pos_dict_dir = "intermediate/word_position_dictionary.json"
    word_position_dict = {}
    with open(word_pos_dict_dir) as f:
        for k,v in json.load(f).items():
            word_position_dict[ v ] = eval(k)

    # take a query gesture, store it
    vector_model = 0  #OR 1
    gestures, gesture_ids = task1.read_vectors(vector_model)

    # find K similar gestures using LSH
    file_name = "intermediate/vectors_dictionary.json"
    vectors = deserialize_vector_file(file_name)
    gestures = list(vectors.keys())
    # print(gestures)
    query = input("Enter query gesture: ")
    query = vectors[query][vector_model]
    input_vector_dimension = len(vectors[gestures[0]][vector_model])
    np.random.seed(0)
    lsh = LSH(num_layers=4, num_hash_per_layer=8, input_dimension=input_vector_dimension)
    lsh.index(vectors)
    results = lsh.query(point=query, t=15, vectors=vectors)
    results = [_[0] for _ in results]
    print(results)

    relevent, not_relevent, res = set(), set(), set()
    res.update(results)
    for m in range(10):
        relevent_gestures = list(map(lambda x: x.strip(), input("Relevent (comma seperated): ").split(","))) # ['1', '22', '20', '31', '2', '4', '6']
        relevent.update(relevent_gestures)
        # relevent_gestures = list(relevent)
        # not_relevent_gestures= list(map(lambda x: x.strip(), input("Not - Relevent (comma seperated): ").split(","))) #['262']
        # not_relevent_gestures = list(res.difference(set(relevent)))
        not_relevent_gestures = list(set(results).difference(set(relevent_gestures)))
        not_relevent.update(not_relevent_gestures)

        not_relevent = not_relevent.difference(relevent)
        # probabilistic relevent feedback
        """
        Unlike above approach, where we transform the query, here we will retrive data according to the 
        feedback we receive from user. So if a gesture is not included in feedback, it will consider it not-relevent.
        Query is not updated here.
        """
        # relevent_gestures_np = np.array([vectors[_][vector_model] for _ in relevent_gestures])
        # not_relevent_gestures_np = np.array([vectors[_][vector_model] for _ in not_relevent_gestures])
        # prob_F_x = lambda x: 1 / (1 + (sum([(lsh.calculate_l2_distance(_, x))**-1 for _ in not_relevent_gestures_np]) / sum([(lsh.calculate_l2_distance(_, x))**-1 for _ in relevent_gestures_np])))

        # candidate_gestures = set(gestures).difference(set(relevent_gestures))
        # probs = {}
        # for gesture in candidate_gestures:
        #     probs[gesture] = prob_F_x(np.array(vectors[gesture][vector_model]))
        # # we take gesture 

        # ########################################################################
        # https://nlp.stanford.edu/IR-book/essir2011/pdf/11prob.pdf
        documents = np.array([[1 if _>0 else 0 for _ in vectors[gesture][vector_model] ] for gesture in gestures])
        relevent_documents = np.array([[1 if _>0 else 0 for _ in vectors[gesture][vector_model]] for gesture in relevent_gestures])
        not_relevent_documents = np.array([[1 if _>0 else 0 for _ in vectors[gesture][vector_model]] for gesture in not_relevent_gestures])
        query_document = np.array([1 if _>0 else 0 for _ in query])

        L = len(relevent_gestures)
        K = len(not_relevent_gestures) + len(relevent_gestures)
        if m==0:
            r0 = [0.5 for i in range(len(query_document))]
            n0 = np.divide(np.sum(documents, axis=0), len(documents))
            r = [0.5 for i in range(len(query_document))]
            n = np.divide(np.sum(documents, axis=0), len(documents))
        else:
            l = relevent_documents.sum(axis=0)
            k = np.divide(l, len(relevent_documents))    
            for i in range(len(query_document)):
                if query_document[i] != 0:
                    r[i] = (l[i] + 0.5) / (L + 1)
                    n[i] = (k[i]-l[i]+0.5 )/(K+L+1)

            l = len(relevent_documents)*np.ones(l.shape) - l
            k = np.divide(l, len(relevent_documents))
            for i in range(len(query_document)):
                if query_document[i] == 0:
                    r0[i] = (l[i] + 0.5) / (L + 1)
                    n0[i] = (k[i]-l[i]+0.5 )/(K+L+1)

        sim = []
        for document in documents:
            c1, c0, cd, count0, count1, countd = 0, 0, 0, 0, 0, 0
            for i in range(len(query_document)):
                if document[i]==1 and query_document[i]==1:
                    # if n[i]==0 or r[i]==1:
                    #     print(i, end=" | ")
                    c1 += (r[i]*(1-n[i])) / (n[i] * (1-r[i]))
                    count1 += 1
                elif document[i]==0 and query_document[i]==0:
                    # if n0[i]==0 or r0[i]==1:
                    #     print(i, end=" | ")
                    c0 += (r0[i]*(1-n0[i])) / (n0[i] * (1-r0[i]))
                    count0 += 1
                else:
                    cd += (n[i]) / (r[i])
                    countd += 1

            c = ((count0*c0)+(count1*c1)-(countd*cd)) / len(document)
            c = c1 
            sim.append(c)
        # print(sim)
        results = list(set([_[1] for _ in sorted(enumerate(gestures), key=lambda x: -sim[x[0]]) ][:15]))
        print(results)
        res.update(results)
        #######################################################################################3
        # # trial, using sensors from feedback gestures
        # # computing importance by counting non-zero values for particular gesture in each gesture
        # gesture_sensor_distribution_rel = {}
        # for gesture_id in relevent_gestures:
        #     gesture = vectors[gesture_id][vector_model]
        #     sensor = {i:0 for i in range(1,21)}
        #     for i, word in enumerate(gesture):
        #         if word != 0:
        #             sensor[int(word_position_dict[i][1])] += 1
        #     print(sensor)
        #     gesture_sensor_distribution_rel[gesture_id] = sensor

        # gesture_sensor_distribution = {}
        # for gesture_id in vectors.keys():
        #     gesture = vectors[gesture_id][vector_model]
        #     sensor = {i:0 for i in range(1,21)}
        #     for i, word in enumerate(gesture):
        #         if word != 0:
        #             sensor[int(word_position_dict[i][1])] += 1
        #     # print(sensor)
        #     gesture_sensor_distribution[gesture_id] = sensor

        # # we have distribution of relevent sensors
        # from scipy.stats import multivariate_normal
        # x = []
        # for gesture_id in gesture_sensor_distribution_rel.keys():
        #     temp = []
        #     for sensor_id in gesture_sensor_distribution_rel[gesture_id].keys():
        #         temp.append(gesture_sensor_distribution_rel[gesture_id][sensor_id])
        #     x.append(temp)
        # x = np.array(x)
        # X = []
        # for gesture_id in gesture_sensor_distribution.keys():
        #     temp = []
        #     for sensor_id in gesture_sensor_distribution[gesture_id].keys():
        #         temp.append(gesture_sensor_distribution[gesture_id][sensor_id])
        #     X.append(temp)
        # X = np.array(X)
        
        # y = multivariate_normal.pdf(X, x.mean(axis=0), np.cov(x, rowvar=False), allow_singular=True)
        # print( [(_[1], y[_[0]]) for _ in sorted(enumerate(gesture_sensor_distribution.keys()), key = lambda x: -y[x[0]])])



if __name__ == "__main__":
    main()