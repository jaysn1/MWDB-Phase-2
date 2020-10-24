#### driver program to run phase 2

import task0a, task0b, task2

def main():
    while True:
        print("""\n\n
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                                                                         │
        │  Phase 2                                                                │
        │                                                                         │
        │    0 - Task 0                                                           │
        │    1 - Task 1                                                           │
        │    2 - Task 2                                                           │
        │    3 - Task 3a                                                          │
        │    4 - Task 3b                                                          │
        │    5 - Task 4a                                                          │
        │    6 - Task 4b                                                          │
        │    7 - Task 4b                                                          │
        │    8 - Task 4b                                                          │
        │                                                                         │
        └─────────────────────────────────────────────────────────────────────────┘""")
        task = int(input("Enter task number: "))
        print("\n\n")
        
        if task == 0:
            print("Executing Task 0a and Task 0b \n")
            task0a.main()  # "load param" method in there takes input parameters
            task0b.main()
        
        elif task == 1:
            print("Executing Task 1 \n")
            print("""
            1 - PCA
            2 - SVD
            3 - NMF
            4 - LDA
            """)
            user_option = int(input("\nEnter method to use to find latent latent semantics: "))

            print("""
                1. TF
                2. TF-IDF
                """)
            vector_model = int(input("Enter a vector model: "))

            k = int(input("Enter k for top-k latent features: "))
            task1.main(user_option, vector_model, k)

            print("""
                {}_transformed_data.json contains the transformed data.
                {}_word_score.txt contains the (word, score) pairs.
                """.format(user_option))

        elif task == 2:
            print("Executing Task 2 \n")
            print("""
            1 - dot product
            2 - PCA
            3 - SVD
            4 - NMF
            5 - LDA
            6 - edit distance
            7 - Dynamic Time Warping
            """)
            user_option = int(input("Enter method to use to find similarity: "))
            gesture_id = input("Enter query gesture: ")

            task2.main(user_option, gesture_id)


if __name__ == "__main__":
    main()