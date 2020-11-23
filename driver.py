"""
driver program for phase3

author: Monil
"""
from RelevanceFeedbackSystemNBModel import ProbabilityFeedbackNBModel
import task1

def main():
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
        inp = int(input("Do you want to create vector model for gestures (0/1)?: "))
        user_option = int(input("\n1:DOT\n 2: PCA\n 3: SVD\n 4: NMF\n 5: LDA\n 6: ED \n7: DTW\nWhat to use for gesture gesture matrix? "))
        if user_option not in [6,7]:
            vector_model = int(input("What vector model to use (TF: 0, TF-IDF:1)?: "))
        
        if inp==1:
            gesture_gesture_similarity = task1.task1_initial_setup(user_option, vector_model, create_vectors=True)
        else:
            gesture_gesture_similarity = task1.task1_initial_setup(user_option, vector_model, create_vectors=False)

        k = int(input("Enter value of k (for outgoing edges): "))
        m = int(input("Enter value of m (for dominant gestures): "))
        n = int(input("Enter numbe of seed nodes: "))
        seed_nodes = []
        for i in range(n):
            seed_nodes.append(input(f"Enter name of seed node {i+1}: "))
        dominant_gestures = task1.main(gesture_gesture_similarity, k, m, seed_nodes, _edge_labels = False)
        print(dominant_gestures)
        

    elif task==2:
        pass
    elif task==3:
        pass
    elif task==4:
        vector_model = int(input("Which vector model to use(0: TF, 1: TF-IDF): "))
        obj = ProbabilityFeedbackNBModel(vector_model=vector_model, num_layers=4, num_hash_per_layer=5)
        print("\nInitial Results: ", [_[0] for _ in obj.initial_query(input("Query: "))])
        
        while(True):  # Until user is satisfied or we do not have anymore results to show
            relevent_ids = list(map(lambda x:x.strip(), input("relevent: ").split(",")))
            notrelevent_ids = list(map(lambda x:x.strip(), input("Not-relevent: ").split(",")))
            results = obj.iteration(relevent_ids, notrelevent_ids)
            print("\nResults: ", [_[0] for _ in results])
            if not input("More results?(y/n)").strip() in ['y', 'Y']:
                break

    elif task==5:
        pass
    elif task==6:
        pass



if __name__=="__main__":
    main()