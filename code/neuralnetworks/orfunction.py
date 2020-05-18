import pandas as pd

#Set weight1, weight2, and bias
Weight1 = 0.1
Weight2 = 0.1
Bias = -0.1

#Inputs and Outputs
TestInputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
CorrectOutputs = [False, True, True, True]
Outputs = []

# Generate and check output
for TestInput, CorrectOutput in zip(TestInputs, CorrectOutputs):
    LinearCombination = TestInput[0] * Weight1 + TestInput[1] * Weight2 + Bias
    Output = int(LinearCombination >= 0)
    IsCorrectString = 'Yes' if Output == CorrectOutput else 'No'
    Outputs.append([TestInput[0], TestInput[1], LinearCombination, Output, IsCorrectString])
    
# Print output
NumWrong = len([Output[4] for Output in Outputs if Output[4] == 'No'])
OutputFrame = pd.DataFrame(Outputs, columns=[
        'Input 1', '  Input 2',  '  Linear Combination', '  Activation Output', '  Is Correct'])

if not NumWrong: 
    print('Nice! You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(NumWrong))
print(OutputFrame.to_string(index = False))