# dtw_similarity
# Uses dynamic time warping to find similarity between two gestures.

"""
int DTWDistance(s: array [1..n], t: array [1..m]) {
    DTW := array [0..n, 0..m]
    
    for i := 0 to n
        for j := 0 to m
            DTW[i, j] := infinity
    DTW[0, 0] := 0
    
    for i := 1 to n
        for j := 1 to m
            cost := d(s[i], t[j])
            DTW[i, j] := cost + minimum(DTW[i-1, j  ],    // insertion
                                        DTW[i  , j-1],    // deletion
                                        DTW[i-1, j-1])    // match
    
    return DTW[n, m]
}
"""

# Generalizing DTW to the multi-dimensional case requires an adaptive approach: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/
# Multi-Dimensional Dynamic Time Warping for GestureRecognition: https://www.researchgate.net/publication/228740947_Multi-dimensional_dynamic_time_warping_for_gesture_recognition


def distance(element1, element2):
    return abs(element1 - element2)

def multi_dimension_dynamic_time_warping(vector1, vector2):
    pass

def dynamic_time_warping(vector1, vector2):
    m, n = len(vector1), len(vector2)
    DTW = [[float("inf") for _ in  range(n+1)] for _ in  range(m+1)]
    
    DTW[0][0] = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = distance(vector1[i-1], vector2[j-1])
            DTW[i][j] = cost + min(DTW[i-1][j],
                               DTW[i][j-1],
                               DTW[i-1][j-1])
    return DTW[m][n]

if __name__=="__main__":
    v1 = [ord(_)-97 for _ in "kitten"]
    print("kitten")
    v2 = [ord(_)-97 for _ in "sitting"]
    print("sitting", dynamic_time_warping(v1, v2))
    
    v2 = [ord(_)-97 for _ in "sitten"]
    print("sitten", dynamic_time_warping(v1, v2))

    v2 = [ord(_)-97 for _ in "kitchen"]
    print("kitchen", dynamic_time_warping(v1, v2))