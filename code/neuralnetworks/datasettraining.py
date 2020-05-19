import numpy as np
import pandas as pd 

# Setting the random seed
np.random.seed(42)

def StepFunction(t):
    if t > 0:
        return 1
    else:
        return 0

def Prediction(X, W, B):
    return StepFunction((np.matmul(X, W) + B)[0])

def PerceptronStep(X1, X2, Y, W, B, LearningRate = 0.01):
    for Index in range(0, len(X1)):
        LinearCombination = X1[Index] * W[0] + X2[Index] * W[1] + B
        Output = int(StepFunction(LinearCombination))
        if Output != Y[Index]:
            W[0] = W[0] + ((Y[Index] - Output) * LearningRate * X1[Index])
            W[1] = W[1] + ((Y[Index] - Output) * LearningRate * X2[Index])
            B = B + ((Y[Index] - Output) * LearningRate)
    return W, B

def TrainPerceptronAlgorithm(X1, X2, Y, LearningRate = 0.01, NumEpochs = 10000):
    Xmin, Xmax= min(X1), max(X1)
    Ymin, Ymax= min(Y), max(Y)
    W = np.array(np.random.rand(2,1))
    B = np.random.rand(1)[0] + Xmax
    # These are the solution lines that get plotted below.
    BoundaryLines = []
    for Count in range(NumEpochs):
        # In each epoch, we apply the perceptron step.
        W, B = PerceptronStep(X1, X2, Y, W, B, LearningRate)
        BoundaryLines.append((-W[0]/W[1], -B/W[1]))
    return W, B
    
    
TrainingDatas = pd.read_csv("res/dataset.csv")
Weights, Bias = TrainPerceptronAlgorithm(TrainingDatas['X1'], TrainingDatas['X2'], TrainingDatas['Label'])

W1 = Weights[0, 0]
W2 = Weights[1, 0]
B = Bias
for Index in range(0, len(TrainDatas['X1'])):
    LinearCombination = TrainDatas['X1'][Index] * W1 + TrainDatas['X2'][Index] * W2 + B
    Output = int(StepFunction(LinearCombination))
    if Output != TrainDatas['Label'][Index]:
        print('fuck you')