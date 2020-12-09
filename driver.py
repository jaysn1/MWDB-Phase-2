"""
driver program for phase3

author: Monil
"""
from RelevanceFeedbackSystemNBModel import ProbabilityFeedbackNBModel
import task1
import task2
import task5
import task6
import lsh
from lsh import *

def main():
    while(True):
        print("""\n
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                                                                         │
        │  Phase 3                                                                │
        │                                                                         │
        │    1 - Task 1                                                           │
        │    2 - Task 2                                                           │
        │    3 - Task 3                                                           │
        │    4 - Task 4                                                           │
        │    5 - Task 5                                                           │
        │    6 - Task 6                                                           │
        │    7 - exit                                                             │
        └─────────────────────────────────────────────────────────────────────────┘""")
        task = int(input("Enter task number: "))
        
        if task == 1:
            # required inputs -> gesture gesture matrix, k, m ,seed nodes
            inp = input("Do you want to create vector model for gestures (y/n)?: ") in ['y', 'Y']
            user_option = int(input("\n 1: DOT\n 2: PCA\n 3: SVD\n 4: NMF\n 5: LDA\nWhat to use for gesture gesture similarity matrix? "))
            if user_option not in [6,7]:
                vector_model = int(input("What vector model to use (TF: 0, TF-IDF:1)?: "))
            else:
                vector_model = 0
            
            if inp:
                gesture_gesture_similarity = task1.task1_initial_setup(user_option, vector_model, create_vectors=True)
            else:
                gesture_gesture_similarity = task1.task1_initial_setup(user_option, vector_model, create_vectors=False)

            k = int(input("Enter value of k (for outgoing edges): "))
            m = int(input("Enter value of m (for dominant gestures): "))
            seed_nodes = list(map(lambda x:x.strip(), input("Enter seed nodes(comma seperated): ").split(",")))
            dominant_gestures = task1.main(gesture_gesture_similarity, k, m, seed_nodes, _edge_labels = False, viz=True)
            print(dominant_gestures)

        elif task==2:
            task2.main()
            pass

        elif task==3:
            lsh.main()
            pass

        elif task==4:
            vector_model = int(input("Which vector model to use(0: TF, 1: TF-IDF): "))
            obj = ProbabilityFeedbackNBModel(vector_model=vector_model, num_layers=4, num_hash_per_layer=5)
            query = input("Query: ")
            t = int(input("Number of results: "))
            results = obj.initial_query(query, t)
            print("\nInitial Results: ", [_[0] for _ in results])
            
            while(True):  # Until user is satisfied or we do not have anymore results to show
                relevent_ids = list(map(lambda x:x.strip(), input("relevent: ").split(",")))
                notrelevent_ids = list(map(lambda x:x.strip(), input("Not-relevent: ").split(",")))
                t = int(input("Number of results: "))
                results = obj.iteration(relevent_ids, notrelevent_ids, t)
                print("\nResults: ", [_[0] for _ in results])
                if not input("More results?(y/n)").strip() in ['y', 'Y']:
                    break

        elif task==5:
            vectors_file_name = "intermediate/vectors_dictionary.json"
            word_position_dictionary_file_name = "intermediate/word_position_dictionary.json"
            component_position_dictionary_file_name = "intermediate/component_position_dictionary.json"
            sensor_position_dictionary_file_name = "intermediate/sensor_position_dictionary.json"

            vectors = task5.deserialize_vector_file(vectors_file_name)
            word_position_dictionary = task5.deserialize_word_position_dictionary(word_position_dictionary_file_name)
            component_position_dictionary = task5.deserialize_component_position_dictionary(component_position_dictionary_file_name)
            sensor_position_dictionary = task5.deserialize_sensor_position_dictionary(sensor_position_dictionary_file_name)
            
            query_gesture = input("Query: ")
            vector_model = int(input("Which vector model to use(0: TF, 1: TF-IDF): "))

            gestures = list(vectors.keys())
            input_vector_dimension = len(vectors[query_gesture][vector_model])

            lsh = LSH(num_layers=4, num_hash_per_layer=8, input_dimension=input_vector_dimension, is_vector_matrix=True, vector_model=vector_model)
            lsh.index(vectors)
            results = lsh.query(point=vectors[query_gesture][vector_model], t=15, vectors=vectors)
            print("\nInitial Results: ", [_[0] for _ in results])
            while(True):  # Until user is satisfied or we do not have anymore results to show
                relevant_gestures = list(map(lambda x:x.strip(), input("relevent: ").split(",")))
                irrelevant_gestures = list(map(lambda x:x.strip(), input("Not-relevent: ").split(",")))
                modified_query_vector = task5.perform_relevance_feeback(query_gesture, vectors, vector_model=vector_model, relevant_gestures=relevant_gestures, irrelevant_gestures=irrelevant_gestures, word_position_dictionary=word_position_dictionary,
                    component_position_dictionary=component_position_dictionary, sensor_position_dictionary=sensor_position_dictionary)
                results = lsh.query(point=modified_query_vector, t=15, vectors=vectors)
                print("\nResults: ", [_[0] for _ in results])
                
                if not input("More results?(y/n)").strip() in ['y', 'Y']:
                    break

        elif task==6:
            task6.main()

        else:
            break


if __name__=="__main__":
    main()