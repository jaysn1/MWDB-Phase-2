# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:52:03 2020

@author: Jaysn
"""

def calculate_similarity(query, gesture_data):
    distance = 0
    for query_value, gesture_value in zip(query, gesture_data):
        distance += abs(query_value - gesture_value)
    if distance == 0:
        return float("inf")
    else:
        return 1/distance
        