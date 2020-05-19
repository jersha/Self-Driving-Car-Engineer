import numpy as np
import math as maths

def Softmax(SMInputs):
    if(len(SMInputs) != 0):
        SMInputs = np.exp(SMInputs)
        Sum = np.sum(SMInputs)
        SMOutputs = np.divide(SMInputs, Sum)
        return SMOutputs
    else:
        print('Invalid input to Softmax function')

arr = [12, 3, 4, 15]
Outputs = Softmax(arr)
for Output in Outputs:
    print(Output)