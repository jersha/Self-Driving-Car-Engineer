import numpy as np

def CrossEntropy(Y, P):
    Probln = np.log(P)
    ProblnMinus = np.log(np.subtract(1, P))
    CE = np.sum(np.add(np.multiply(Y, Probln), np.multiply(np.subtract(1, Y), ProblnMinus)))
    return -CE

array1 = [1, 1, 0]
array2 = [0.8, 0.7, 0.1]
print(CrossEntropy(array1, array2))