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

from Phase2 import task1
import numpy as np
from lsh import deserialize_vector_file, LSH

def update_query(query, gestures, relevent_gestures, not_relevent_gestures, alpha=1, beta=1):
    pass

def main():
    """
    This function uses gesture representation from task1 in phase 2 as source for gestures.
    It will ask for a query gesture initially and show the results from nearest neigbor (LSH).
    Then, it will ask user to provide some relevent and non-revelent results from it,
    and it will refine query using "Rocchio algorithm" to create probabilistic feedback
    """
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
    np.seed(0)
    lsh = LSH(num_layers=4, num_hash_per_layer=8, input_dimension=input_vector_dimension)
    lsh.index(vectors)

    for m in range(10):    
        results = lsh.query(point=query, t=15, vectors=vectors)
        print([_[0] for _ in results])

        relevent_gestures = list(map(lambda x: x.strip(), input("Relevent (comma seperated): ").split(",")))
        not_relevent_gestures = list(map(lambda x: x.strip(), input("Not - Relevent (comma seperated): ").split(",")))

        # vector space based relevent feedback
        # relevent_gestures_np = np.array([vectors[_][vector_model] for _ in relevent_gestures])
        # centroid_of_relevent = relevent_gestures_np.sum(axis=0) / len(relevent_gestures)

        # not_relevent_gestures_np = np.array([vectors[_][vector_model] for _ in not_relevent_gestures])
        # centroid_of_not_relevent = not_relevent_gestures_np.sum(axis=0) / len(not_relevent_gestures)
        
        # query = query + centroid_of_relevent - centroid_of_not_relevent
        ########################################################################

        # probabilistic relevent feedback
        """
        Unlike above approach, where we transform the query, here we will retrive data according to the 
        feedback we receive from user. So if a gesture is not included in feedback, it will consider it not-relevent.
        Query is not updated here.
        """
        relevent_gestures_np = np.array([vectors[_][vector_model] for _ in relevent_gestures])
        not_relevent_gestures_np = np.array([vectors[_][vector_model] for _ in not_relevent_gestures])
        prob_F_x = lambda x: 1 / (1 + (sum([(lsh.calculate_l2_distance(_, x))**-1 for _ in not_relevent_gestures_np]) / sum([(lsh.calculate_l2_distance(_, x))**-1 for _ in relevent_gestures_np])))

        candidate_gestures = set(gestures).difference(set(relevent_gestures))
        probs = {}
        for gesture in candidate_gestures:
            probs[gesture] = prob_F_x(np.array(vectors[gesture][vector_model]))
        # we take gesture 

        ########################################################################
    

if __name__ == "__main__":

    main()