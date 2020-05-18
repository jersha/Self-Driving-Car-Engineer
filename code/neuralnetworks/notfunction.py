import pandas as pd

#Set weight1, weight2, and bias
Weight = -0.1
Bias = 0.0

#Inputs and Outputs
TestInputs = [0, 1]
CorrectOutputs = [True, False]
Outputs = []

# Generate and check output
for TestInput, CorrectOutput in zip(TestInputs, CorrectOutputs):
    LinearCombination = TestInput * Weight + Bias
    Output = int(LinearCombination >= 0)
    IsCorrectString = 'Yes' if Output == CorrectOutput else 'No'
    Outputs.append([TestInput, LinearCombination, Output, IsCorrectString])
    
# Print output
NumWrong = len([Output[3] for Output in Outputs if Output[3] == 'No'])
OutputFrame = pd.DataFrame(Outputs, columns=[
        'Input', '  Linear Combination', '  Activation Output', '  Is Correct'])

if not NumWrong: 
    print('Nice! You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(NumWrong))
print(OutputFrame.to_string(index = False))