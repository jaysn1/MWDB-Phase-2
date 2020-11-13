#### driver program to run phase 2

import task0a, task0b, task1, task2, task3, task4c, task4d, task4ab
import json

def main():
    while True:
        print("""\n
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Phase 2                                                                │
    │                                                                         │
    │    0 - Task 0                                                           │
    │    1 - Task 1                                                           │
    │    2 - Task 2                                                           │
    │    3 - Task 3                                                           │
    │    4 - Task 4a                                                          │
    │    5 - Task 4b                                                          │
    │    6 - Task 4c                                                          │
    │    7 - Task 4d                                                          │
    │    8 - exit                                                             │
    └─────────────────────────────────────────────────────────────────────────┘""")
        task = int(input("Enter task number: "))
        
        if task == 0:
            print("Executing Task 0a and Task 0b \n")
            task0a.main()  # "load param" method in there takes input parameters
            task0b.main()
            print("Task 0a & 0b completed. \n\t'.wrd' files & TF, TF-IDF vectors will be present in the data directory.")
        
        elif task == 1:
            print("Executing Task 1")
            print("""
            1 - PCA
            2 - SVD
            3 - NMF
            4 - LDA""")
            user_option = int(input("Enter method to use to find latent latent semantics: "))

            print("""
                1. TF
                2. TF-IDF""")
            vector_model = int(input("Enter a vector model: "))

            k = int(input("Enter k for top-k latent features: "))
            try:
                task1.main(user_option, vector_model, k)
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))

        elif task == 2:
            print("Executing Task 2")
            print("""
            1 - dot product
            2 - PCA
            3 - SVD
            4 - NMF
            5 - LDA
            6 - edit distance
            7 - Dynamic Time Warping""")
            user_option = int(input("Enter method to use to find similarity: "))
            gesture_id = input("Enter query gesture: ")

            try:
                topk = task2.main(user_option, gesture_id)
                print("Top 10 gestures similar to {}: {}".format(gesture_id, topk))
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))

        elif task == 3:
            print("""
            1 - Dot Product
            2 - PCA
            3 - SVD
            4 - NMF
            5 - LDA
            6 - Edit Distance
            7 - DTW Distance""")
            user_option = int(input("Enter method to use to find gesture-gesture similarity: "))
            print("""
            1 - SVD (task 3a)
            2 - NMF(task 3b)""")
            latent_semantics_option = int(input("Enter method to find top-p latent semantics: "))

            p = int(input("Enter the desired p-value (for top-p principal components): "))

            try:
                task3.main(user_option, p, latent_semantics_option)
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))

        elif task == 4:
            try:
                task4ab.main(1)
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))
            
            
        elif task == 5:
            try:
                task4ab.main(2)
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))
        
        elif task == 6:
            print("""
            1 - Dot Product
            2 - PCA
            3 - SVD
            4 - NMF
            5 - LDA
            6 - Edit Distance
            7 - DTW Distance""")
            user_option = int(input("Enter which type of gesture gesture similarity to use: "))
    
            try:
                task4c.main(user_option)
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))        
            
        elif task == 7:
            print("""
            1 - Dot Product
            2 - PCA
            3 - SVD
            4 - NMF
            5 - LDA
            6 - Edit Distance
            7 - DTW Distance""")
            user_option = int(input("Enter which type of gesture gesture similarity to use: "))
    
            try:
                task4d.main(user_option)
            except ValueError as e:
                print("\n\n" + "="*(len(str(e))+12))
                print("="*5, str(e), "="*5)
                print("="*(len(str(e))+12))  
            

        else:
            break



if __name__ == "__main__":
    main()