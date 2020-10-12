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


def distance(element1, element2):
    return abs(element1 - element2)**2


def multi_dimension_dynamic_time_warping(gesture1, gesture2, avg_dict):
    """
    # Generalizing DTW to the multi-dimensional case requires an adaptive approach: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/
    # Multi-Dimensional Dynamic Time Warping for GestureRecognition: https://www.researchgate.net/publication/228740947_Multi-dimensional_dynamic_time_warping_for_gesture_recognition
    input gestures are assumed to be dictionaries having component->sensor->time_series
    """
    assert sorted(gesture1.keys())==sorted(gesture2.keys())

    Q, C = [], []
    weight1, weight2 = [], []
    for component_id in gesture1:
        for sensor_id in gesture1[component_id]:
            Q.append(gesture1[component_id][sensor_id])
            C.append(gesture2[component_id][sensor_id])
            weight1.append(avg_dict[0][component_id][sensor_id][0])
            weight2.append(avg_dict[1][component_id][sensor_id][0])
    M = len(Q)
    m, n = len(Q[0]), len(C[0])
    DTW = [[float("inf") for _ in  range(n+1)] for _ in  range(m+1)]
    DTW[0][0] = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = sum([distance(Q[_][i-1], C[_][j-1])*abs(weight1[_] - weight2[_]) for _ in range(M)])

            DTW[i][j] = cost + min(DTW[i-1][j],
                               DTW[i][j-1],
                               DTW[i-1][j-1])
    return DTW[m][n]/(m+n)


def derivative_dynamic_time_wraping(vector1, vector2, weight=1):
    """
    # Derivative Dynamic Time Warping: https://www.ics.uci.edu/~pazzani/Publications/sdm01.pdf
    """
    if len(vector1)>=3:
        derivateve_vector1 = [vector1[0]] + [(vector1[i+1]-vector1[i-1])/2 for i in range(1, len(vector1)-1)] + [vector1[-1]]
    else:
        derivateve_vector1 = vector1
    if len(vector2)>=3:
        derivateve_vector2 = [vector2[0]] + [(vector2[i+1]-vector2[i-1])/2 for i in range(1, len(vector2)-1)] + [vector2[-1]]
    else:
        derivateve_vector2 = vector2
    return dynamic_time_warping(derivateve_vector1, derivateve_vector2, weight)


def dynamic_time_warping(vector1, vector2, weight=1):
    """
    # have to consider distance. so normalising by length
    """
    m, n = len(vector1), len(vector2)
    DTW = [[float("inf") for _ in  range(n+1)] for _ in  range(m+1)]
    
    DTW[0][0] = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = distance(vector1[i-1], vector2[j-1])
            DTW[i][j] = cost + min(DTW[i-1][j],
                               DTW[i][j-1],
                               DTW[i-1][j-1])
    return DTW[m][n]*weight/(m+n)

if __name__=="__main__":
    # v1 = [ord(_)-97 for _ in "kitten"]
    # print("kitten")
    # v2 = [ord(_)-97 for _ in "sitting"]
    # print("sitting", dynamic_time_warping(v1, v2))
    
    # v2 = [ord(_)-97 for _ in "sitten"]
    # print("sitten", dynamic_time_warping(v1, v2))

    # v2 = [ord(_)-97 for _ in "kitchen"]
    # print("kitchen", dynamic_time_warping(v1, v2))

    gesture1 = {
        'X': {'1': [0,0,0,0], '2': [-1, -0.5, 0, 1], '3':[-1,-1,1,1]},
        'Y': {'1': [1,-1,1,-1], '2': [-1,-1,1,1], '3':[-1, 0.5, 0.5, 1]},
        'Z': {'1': [1,0.5,0.5,-1], '2': [0,0,0,0], '3':[1,1,0.5,-1]}
    }
    gesture2 = {
        'X': {'1': [0,1,-1,0], '2': [-1, -0.5, 0, 1], '3':[-1,-1,1,1]},
        'Y': {'1': [-1,1,-1,-1], '2': [-1,-1,1,1], '3':[-1, 0.8, 0.9, 1]},
        'Z': {'1': [1,0.7,0.7,-1], '2': [1,0.1,0.3,-1], '3':[-1,-1,0.5,1]}
    }

    print("Multi DTW: ",multi_dimension_dynamic_time_warping(gesture1, gesture2))