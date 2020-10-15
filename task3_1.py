# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:56:53 2020

@author: Jaysn
"""
import json
from similarity_calculators.dot_product_similarity import dot_product_similarity


def main():
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  Phase 2 -Task 2                                                        │
  │                                                                         │
  │                                                                         │
  │    1 - Dot Product                                                      │
  │    2 - PCA                                                              │
  │    3 - SVD                                                              │
  │    4 - NMF                                                              │
  │    5 - LDA                                                              │
  │    6 - Edit Distance                                                    │
  │    7 - DTW Distance                                                     │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘""")
    user_option = int(input("\nEnter method to use to find similarity: "))
    
    vectors_dir="intermediate/vectors_dictionary.json"
    with open(vectors_dir) as f:
        vectors = json.load(f)
    
    if user_option == 1:
        vector_model = int(input("Enter vector model to use? (1: TF, 2: TF-IDF): "))
        gesture_gesture_similarity = []
        for gesture_id in vectors:
            vector = vectors[gesture_id][vector_model-1]
            gesture_similarity = []
            for gesture in vectors:
                gesture_similarity.append(dot_product_similarity(vector,vectors[gesture][vector_model-1]))
            gesture_gesture_similarity.append(gesture_similarity)
        return gesture_gesture_similarity

a = main()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            